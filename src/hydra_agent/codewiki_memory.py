"""Resumable indexing with per-source retries for the CodeWikiBench corpus."""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .bench_data import atomic_json, digest, read_json
from .hydradb import HydraError, HydraMemory
from .indexing import Corpus

STALL_WARNING_SECONDS = 300


class IndexRetryExhausted(HydraError):
    """A source reached its persistent upload-attempt limit."""


class CodeWikiMemory(HydraMemory):
    """Preserve source filters while respecting the hosted API's 200-ID query limit."""

    def configure_search(self, path, *, max_queries=120):
        """Persist logical query reservations and successful exact-query results."""
        if type(max_queries) is not int or max_queries < 0:
            raise ValueError("Search budget must be nonnegative; 0 disables the cap")
        identity = digest(
            {
                "corpus": self.corpus.manifest(),
                "database": self.config.database,
                "collection": self.collection,
                "adapter": "codewiki_search_cache_v1",
            }
        )
        saved = (
            read_json(path)
            if path.exists()
            else {
                "identity": identity,
                "queries": self.search_calls,
                "results": {},
            }
        )
        if saved["identity"] != identity:
            raise ValueError("Search cache belongs to different sources or database")
        self.search_path, self.search_state = path, saved
        self.max_queries, self.search_calls = max_queries, saved["queries"]
        if saved.get("limit") != max_queries:
            saved.setdefault("limit_history", []).append(max_queries)
        saved["limit"] = max_queries
        atomic_json(path, saved)

    def search(self, query, *, scope, limit):
        if scope != self.corpus.scope or not self.ready:
            raise HydraError("Repository graph is not ready or scope does not match")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search requires a nonempty question")
        eligible = [s for s in self.sources.values() if s.path not in self.dirty_paths]
        if not eligible:
            return []
        limit = max(1, min(limit, 8))
        cache = getattr(self, "search_state", None)
        key = digest(
            {"query": query.strip(), "limit": limit, "ids": sorted(s.id for s in eligible)}
        )
        if cache is not None and key in cache["results"]:
            saved = cache["results"][key]
            if digest(saved["hits"]) != saved["digest"]:
                raise ValueError("Cached retrieval evidence changed")
            self.report(
                "retrieval_cache_hit", hits=len(saved["hits"]), search_calls=self.search_calls
            )
            return saved["hits"]
        cap = getattr(self, "max_queries", 40)
        if cap and self.search_calls >= cap:
            raise HydraError("HydraDB search-call limit reached")
        self.search_calls += 1
        if cache is not None:
            # Reserve before the network call: failed/interrupted queries still count.
            cache["queries"] = self.search_calls
            atomic_json(self.search_path, cache)
        hits = []
        for offset in range(0, len(eligible), 200):
            shard = HydraMemory(
                self.config,
                Corpus(scope, eligible[offset : offset + 200], []),
                client=self.client,
                existing_collection=self.collection,
                report=self.report,
            )
            shard.ready = True
            hits.extend(shard.search(query, scope=scope, limit=limit))
        # Client score merge, not a second global rerank. Never omit source IDs.
        hits.sort(
            key=lambda h: h.get("score") if isinstance(h.get("score"), (int, float)) else 0,
            reverse=True,
        )
        self.report(
            "codewiki_query_merge",
            batches=(len(eligible) + 199) // 200,
            candidates=len(hits),
            search_calls=self.search_calls,
        )
        hits = hits[:limit]
        if cache is not None:
            cache["results"][key] = {"hits": hits, "digest": digest(hits)}
            atomic_json(self.search_path, cache)
        return hits


def index_corpus(
    config,
    corpus,
    manifest_path,
    *,
    timeout=1200,
    workers=4,
    max_attempts=2,
    max_failures=0,
    report=None,
    client=None,
):
    if not 1 <= workers <= 8:
        raise ValueError("Index workers must be between 1 and 8")
    if max_attempts < 1:
        raise ValueError("Index upload attempts must be positive")
    if max_failures < 0:
        raise ValueError("Index max failures cannot be negative")
    # HTTP requests can run concurrently; serialize trace writes and terminal callbacks.
    report_lock = Lock()

    def serialized_report(kind, **data):
        if report:
            with report_lock:
                report(kind, **data)

    memory = CodeWikiMemory(config, corpus, report=serialized_report, client=client, poll_seconds=5)
    expected = {**corpus.manifest(), "database": config.database, "collection": memory.collection}
    manifest = (
        read_json(manifest_path)
        if manifest_path.exists()
        else {
            **expected,
            "status": "pending",
            "attempts": {},
            "statuses": {},
        }
    )
    for key in ("sources", "scope", "database", "collection"):
        if manifest.get(key) != expected[key]:
            memory.close()
            raise ValueError("Saved HydraDB index does not match corpus/configuration")
    attempts, statuses = manifest["attempts"], manifest["statuses"]
    # Sources that permanently failed indexing but were tolerated under the skip budget.
    skipped = manifest.setdefault("skipped", {})
    deadline = time.monotonic() + timeout

    def save():
        atomic_json(manifest_path, manifest)

    def batches(items, request, accept, *, before=None):
        # At most `workers` batches are submitted at once. Checkpoints are owned by
        # this thread; workers only perform HTTP calls and validate response IDs.
        if not items:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(items), workers * 20):
                group = [
                    items[i : i + 20]
                    for i in range(offset, min(offset + workers * 20, len(items)), 20)
                ]
                if before:
                    before(group)
                futures = [pool.submit(request, batch) for batch in group]
                error = None
                for future in as_completed(futures):
                    try:
                        accept(future.result())
                    except Exception as exc:  # noqa: BLE001 -- retain other acknowledged batches
                        if error is None:
                            error = exc
                if error is not None:
                    raise error

    def status_batch(batch):
        params = [("database", config.database), ("collection", memory.collection)]
        params += [("ids", sid) for sid in batch]
        data = memory._request("GET", "/context/status", params=params, deadline=deadline)
        found = {}
        for item in data.get("statuses", []):
            sid = item.get("id")
            if sid not in batch or sid in found:
                raise HydraError("Unexpected or duplicate indexing status ID")
            found[sid] = item
        if set(found) != set(batch):
            raise HydraError("Incomplete indexing status response; refusing to assume readiness")
        return found

    def refresh(ids):
        try:
            batches(ids, status_batch, statuses.update)
        finally:
            # One checkpoint per scan, including partial results on failure, instead
            # of rewriting the entire corpus manifest after every 20-source response.
            save()

    def reserve_uploads(group):
        for batch in group:
            for source in batch:
                attempts[source.id] = attempts.get(source.id, 0) + 1
        save()

    def upload_batch(batch):
        items = [
            {
                "id": s.id,
                "database": config.database,
                "collection": memory.collection,
                "title": s.path,
                "type": "document",
                "content": {"text": s.text},
                "additional_metadata": {
                    "path": s.path,
                    "base_commit": corpus.scope.base_commit,
                    "sha256": s.sha256,
                },
            }
            for s in batch
        ]
        fields = {
            "database": config.database,
            "collection": memory.collection,
            "type": "knowledge",
            "upsert": "true",
            "app_knowledge": json.dumps(items),
        }
        data = memory._request(
            "POST",
            "/context/ingest",
            deadline=deadline,
            files={k: (None, v) for k, v in fields.items()},
        )
        results = data.get("results", [])
        if len(results) != len(batch) or {r.get("id") for r in results} != {s.id for s in batch}:
            raise HydraError("Ingestion did not acknowledge every source")
        if any(
            result.get(k) for result in results for k in ("error", "error_code", "relations_error")
        ):
            raise HydraError("HydraDB rejected a source")
        return results

    acknowledged = set(statuses)

    def accept_upload(results):
        for result in results:
            sid = result["id"]
            statuses[sid] = {"id": sid, "indexing_status": "pending"}
            acknowledged.add(sid)
        save()
        memory.report("codewiki_upload", uploaded=len(acknowledged), total=len(corpus.sources))

    try:
        save()
        memory.report("codewiki_index_workers", workers=workers, max_attempts=max_attempts)
        memory.report("codewiki_database", message="Checking HydraDB readiness")
        params = {"database": config.database}
        infra = memory._request(
            "GET", "/databases/status", params=params, allow_status=(404,), deadline=deadline
        )
        if infra.get("_http_status") == 404:
            memory.report("codewiki_database", message="Creating HydraDB database")
            memory._request(
                "POST", "/databases", json=params, allow_status=(409,), deadline=deadline
            )
        if infra.get("infra", {}).get("ready_for_ingestion") is not True:
            memory.report("codewiki_database", message="Waiting for database provisioning")
        while infra.get("infra", {}).get("ready_for_ingestion") is not True:
            memory._pause(deadline)
            infra = memory._request("GET", "/databases/status", params=params, deadline=deadline)
        manifest["infrastructure"] = infra
        memory.report("codewiki_database", message="HydraDB ready for ingestion")
        # Revalidate even previously completed sources; deleted remote data cannot pass resume.
        previous = [s.id for s in corpus.sources if attempts.get(s.id, 0)]
        if previous:
            memory.report("codewiki_index_resume", sources=len(previous))
        refresh(previous)
        acknowledged.update(statuses)
        last_completed = sum(s.get("indexing_status") == "completed" for s in statuses.values())
        last_progress = last_warning = time.monotonic()
        while True:
            pending = [
                s
                for s in corpus.sources
                if statuses.get(s.id, {}).get("indexing_status") != "completed"
                and s.id not in skipped
            ]
            if not pending:
                memory.report(
                    "codewiki_indexing",
                    completed=len(corpus.sources) - len(skipped),
                    total=len(corpus.sources),
                )
                break
            upload = []
            for source in pending:
                state = statuses.get(source.id, {}).get("indexing_status")
                if not attempts.get(source.id) or state in ("failed", "errored", "not_found"):
                    if attempts.get(source.id, 0) >= max_attempts:
                        code = statuses.get(source.id, {}).get("error_code", "unknown")
                        if len(skipped) < max_failures:
                            skipped[source.id] = {
                                "path": source.path,
                                "error_code": code,
                                "attempts": attempts.get(source.id, 0),
                            }
                            save()
                            memory.report(
                                "codewiki_index_skip",
                                path=source.path,
                                error_code=code,
                                skipped=len(skipped),
                                allowed=max_failures,
                            )
                            continue
                        raise IndexRetryExhausted(
                            f"Source {source.path} failed indexing after {attempts[source.id]} "
                            f"upload attempts ({code}); limit {max_attempts} reached and skip "
                            f"budget {max_failures} exhausted"
                        )
                    upload.append(source)
            retries = sum(bool(attempts.get(source.id)) for source in upload)
            if retries:
                memory.report("codewiki_retry", sources=retries)
            batches(upload, upload_batch, accept_upload, before=reserve_uploads)
            refresh([s.id for s in pending])
            complete = sum(s.get("indexing_status") == "completed" for s in statuses.values())
            now = time.monotonic()
            if complete > last_completed:
                last_progress = now
            last_completed = complete
            remaining = [s for s in pending if statuses[s.id].get("indexing_status") != "completed"]
            states = dict(
                Counter(statuses[s.id].get("indexing_status", "unknown") for s in remaining)
            )
            memory.report(
                "codewiki_indexing", completed=complete, total=len(corpus.sources), states=states
            )
            if remaining and now - max(last_progress, last_warning) >= STALL_WARNING_SECONDS:
                memory.report(
                    "codewiki_index_stalled",
                    seconds=int(now - last_progress),
                    remaining=len(remaining),
                    sources=[
                        {"path": s.path, "status": statuses[s.id].get("indexing_status", "unknown")}
                        for s in remaining[:5]
                    ],
                )
                last_warning = now
            if complete < len(corpus.sources):
                memory._pause(deadline)
        manifest["status"] = "completed"
        manifest.pop("error", None)
        save()
        # Exclude permanently-failed sources from retrieval; they remain readable via read_file.
        memory.dirty_paths.update(entry["path"] for entry in skipped.values())
        memory.ready = True
        return memory
    except BaseException as exc:
        manifest["status"] = "incomplete"
        manifest["error"] = str(exc) if isinstance(exc, HydraError) else type(exc).__name__
        save()
        memory.close()
        raise


def open_index(config, corpus: Corpus, manifest_path, report=None):
    saved = read_json(manifest_path)
    if saved.get("status") != "completed" or saved.get("sources") != corpus.manifest()["sources"]:
        raise ValueError("Index is incomplete or sources changed; run index first")
    if (
        saved["database"] != config.database
        or saved["collection"] != "attempt_" + corpus.scope.attempt_id
    ):
        raise ValueError("Index database/collection mismatch")
    memory = CodeWikiMemory(config, corpus, existing_collection=saved["collection"], report=report)
    # Skipped sources are not present remotely; exclude them from retrieval and readiness checks.
    memory.dirty_paths.update(entry["path"] for entry in saved.get("skipped", {}).values())
    try:
        memory.prepare(timeout=300)
    except BaseException:
        memory.close()
        raise
    return memory
