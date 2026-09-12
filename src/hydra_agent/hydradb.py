"""Hosted HydraDB v2 ingestion and scoped retrieval; API key stays in the host controller."""

import json
import time
from collections.abc import Callable

import httpx

from .config import HydraConfig
from .indexing import Corpus
from .memory import MemoryScope


class HydraError(RuntimeError):
    """A sanitized API/protocol failure safe to put in run logs."""


class HydraMemory:
    def __init__(
        self,
        config: HydraConfig,
        corpus: Corpus,
        *,
        client: httpx.Client | None = None,
        report: Callable | None = None,
        poll_seconds: float = 2,
        existing_collection: str | None = None,
    ):
        self.config = config
        self.corpus = corpus
        self.retrieval_only = existing_collection is not None
        if existing_collection is not None and not existing_collection.strip():
            raise ValueError("Existing collection must not be empty")
        self.collection = existing_collection or "attempt_" + corpus.scope.attempt_id
        self.sources = {source.id: source for source in corpus.sources}
        self.client = client or httpx.Client(
            base_url=config.base_url, timeout=30, follow_redirects=False
        )
        self.report = report or (lambda *args, **kwargs: None)
        self.poll_seconds = poll_seconds
        self.dirty_paths: set[str] = set()
        self.ready = False
        self.search_calls = 0

    def _request(
        self,
        method: str,
        path: str,
        *,
        deadline: float | None = None,
        allow_status: tuple[int, ...] = (),
        **kwargs,
    ) -> dict:
        if self.retrieval_only and (method, path) not in {
            ("GET", "/context/status"),
            ("POST", "/query"),
        }:
            raise HydraError("Retrieval-only memory forbids database creation and ingestion")
        for attempt in range(3):
            remaining = deadline - time.monotonic() if deadline is not None else 30
            if remaining <= 0:
                raise HydraError("HydraDB indexing deadline exceeded")
            started = time.monotonic()
            response = self.client.request(
                method,
                path,
                headers={"Authorization": "Bearer " + self.config.api_key, "API-Version": "2"},
                timeout=min(30, remaining),
                **kwargs,
            )
            self.report(
                "hydradb_request",
                method=method,
                path=path,
                status=response.status_code,
                retry=attempt,
                elapsed_seconds=round(time.monotonic() - started, 3),
                request_id=response.headers.get("X-Request-ID"),
            )
            if response.status_code in allow_status:
                return {"_http_status": response.status_code}
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                delay = 2**attempt
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                time.sleep(delay)
                continue
            if not response.is_success:
                raise HydraError(f"HydraDB {path} returned HTTP {response.status_code}")
            envelope = response.json()
            if envelope.get("success") is not True or not isinstance(envelope.get("data"), dict):
                raise HydraError("HydraDB response envelope was unsuccessful or malformed")
            return envelope["data"]
        raise HydraError("HydraDB retry limit exceeded")

    def _pause(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HydraError("HydraDB indexing deadline exceeded")
        time.sleep(min(self.poll_seconds, remaining))

    def prepare(self, *, timeout: float = 900) -> dict:
        if timeout <= 0:
            raise ValueError("Indexing timeout must be positive")
        if self.retrieval_only:
            return self._prepare_existing(timeout=timeout)
        started = time.monotonic()
        deadline = started + timeout
        params = {"database": self.config.database}
        status = self._request(
            "GET", "/databases/status", params=params, deadline=deadline, allow_status=(404,)
        )
        if status.get("_http_status") == 404:
            self._request("POST", "/databases", json=params, deadline=deadline, allow_status=(409,))
        while status.get("infra", {}).get("ready_for_ingestion") is not True:
            self._pause(deadline)
            status = self._request("GET", "/databases/status", params=params, deadline=deadline)
        source_list = list(self.sources.values())
        for offset in range(0, len(source_list), 20):
            batch = source_list[offset : offset + 20]
            items = [
                {
                    "id": source.id,
                    "database": self.config.database,
                    "collection": self.collection,
                    "title": source.path,
                    "type": "document",
                    "content": {"text": source.text},
                    "additional_metadata": {
                        "path": source.path,
                        "base_commit": self.corpus.scope.base_commit,
                        "sha256": source.sha256,
                    },
                }
                for source in batch
            ]
            # OpenAPI requires multipart, including for pre-extracted text.
            fields = {
                "database": self.config.database,
                "collection": self.collection,
                "type": "knowledge",
                "upsert": "true",
                "app_knowledge": json.dumps(items),
            }
            data = self._request(
                "POST",
                "/context/ingest",
                deadline=deadline,
                files={key: (None, value) for key, value in fields.items()},
            )
            expected = {source.id for source in batch}
            results = data.get("results", [])
            if (
                {item.get("id") for item in results} != expected
                or len(results) != len(batch)
                or any(
                    item.get("error")
                    or item.get("error_code")
                    or item.get("relations_error")
                    or item.get("status") in ("failed", "errored")
                    for item in results
                )
            ):
                raise HydraError("HydraDB did not accept every source in the ingestion batch")
            self.report("hydradb_ingested", count=offset + len(batch), total=len(source_list))

        pending = set(self.sources)
        while pending:
            ordered = sorted(pending)
            for offset in range(0, len(ordered), 20):
                batch_ids = ordered[offset : offset + 20]
                params = [("database", self.config.database), ("collection", self.collection)]
                params.extend(("ids", source_id) for source_id in batch_ids)
                data = self._request("GET", "/context/status", params=params, deadline=deadline)
                for item in data.get("statuses", []):
                    if item.get("id") not in batch_ids:
                        raise HydraError("Unexpected source ID in indexing status")
                    if item.get("indexing_status") in ("failed", "errored"):
                        raise HydraError("HydraDB source indexing failed; inspect request IDs")
                    # graph_creation is searchable but is NOT a completed graph.
                    if item.get("indexing_status") == "completed":
                        pending.discard(item["id"])
            self.report("hydradb_indexing", pending=len(pending), total=len(self.sources))
            if pending:
                self._pause(deadline)
        self.ready = True
        return {
            "database": self.config.database,
            "collection": self.collection,
            "source_count": len(self.sources),
            "status": "completed",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    def _prepare_existing(self, *, timeout: float) -> dict:
        """Read-only readiness check; never create, upload, or repair missing remote data."""
        self.ready = False
        started = time.monotonic()
        ids = sorted(self.sources)
        if not ids:
            raise HydraError("Cannot reuse an empty source allowlist")
        for offset in range(0, len(ids), 20):
            batch = ids[offset : offset + 20]
            params = [("database", self.config.database), ("collection", self.collection)]
            params.extend(("ids", source_id) for source_id in batch)
            data = self._request(
                "GET", "/context/status", params=params, deadline=started + timeout
            )
            statuses = data.get("statuses", [])
            if (
                len(statuses) != len(batch)
                or {item.get("id") for item in statuses} != set(batch)
                or any(item.get("indexing_status") != "completed" for item in statuses)
            ):
                raise HydraError(
                    "Existing graph has missing or incomplete sources; no ingestion attempted"
                )
        self.ready = True
        self.report("hydradb_reused", collection=self.collection, source_count=len(ids))
        return {
            "database": self.config.database,
            "collection": self.collection,
            "source_count": len(ids),
            "status": "completed",
            "reused": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    def mark_changed(self, paths: set[str]) -> None:
        self.dirty_paths.update(paths)

    def refresh(self, workspace) -> None:
        try:
            self.mark_changed(workspace.changed_paths())
        except Exception:  # noqa: BLE001 -- prevent retrieval when freshness cannot be verified
            self.ready = False
            raise HydraError(
                "Could not verify workspace freshness; graph retrieval disabled"
            ) from None

    def search(self, query: str, *, scope: MemoryScope, limit: int) -> list[dict]:
        if scope != self.corpus.scope:
            raise HydraError("Memory scope does not match the indexed repository attempt")
        if not self.ready:
            raise HydraError("Repository graph is not ready")
        eligible = {
            key for key, source in self.sources.items() if source.path not in self.dirty_paths
        }
        if not eligible:
            return []  # Empty ids would widen the API search to the whole collection.
        if self.search_calls >= 40:
            raise HydraError("HydraDB search-call limit reached")
        self.search_calls += 1
        limit = max(1, min(limit, 8))
        data = self._request(
            "POST",
            "/query",
            json={
                "database": self.config.database,
                "collections": [self.collection],
                "ids": sorted(eligible),
                "query": query[:4000],
                "type": "knowledge",
                "query_by": "hybrid",
                "mode": "thinking",
                "graph_context": True,
                "query_forceful_relations": True,
                "recency_bias": 0,
                "temporal_reasoning": False,
                "max_results": limit,
            },
        )
        candidates = [(chunk, "ranked") for chunk in (data.get("chunks") or [])]
        candidates.extend(
            (chunk, "related") for chunk in (data.get("additional_context") or {}).values()
        )
        hits, seen = [], set()
        for chunk, origin in candidates:
            source_id = chunk.get("id")
            if source_id not in eligible:
                continue
            collection = chunk.get("collection") or chunk.get("sub_tenant_id")
            if collection is not None and collection != self.collection:
                continue
            text = chunk.get("chunk_content") or ""
            identity = (source_id, chunk.get("chunk_uuid") or text)
            if identity in seen or not text:
                continue
            seen.add(identity)
            source = self.sources[source_id]
            hits.append(
                {
                    "source_id": source_id,
                    "path": source.path,
                    "base_commit": scope.base_commit,
                    "sha256": source.sha256,
                    "text": text.encode()[:1400].decode("utf-8", errors="replace"),
                    "truncated": len(text.encode()) > 1400,
                    "score": chunk.get("relevancy_score"),
                    "origin": origin,
                    "chunk_id": chunk.get("chunk_uuid"),
                    "instruction": "Base snapshot evidence. Read current file before editing.",
                }
            )
            if len(hits) == limit:
                break
        # Raw graph paths may reference chunks outside the allowed source set. Do not inject them.
        self.report(
            "hydradb_search",
            hits=len(hits),
            candidates=len(candidates),
            excluded_changed_files=len(self.dirty_paths),
            search_calls=self.search_calls,
        )
        return hits

    def close(self) -> None:
        self.client.close()
