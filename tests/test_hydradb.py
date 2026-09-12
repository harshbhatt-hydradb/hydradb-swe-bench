import io
import json
import tarfile
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace

import httpx
import pytest

from hydra_agent.config import HydraConfig
from hydra_agent.hydradb import HydraError, HydraMemory
from hydra_agent.indexing import build_corpus, reuse_corpus
from hydra_agent.memory import MemoryScope


@pytest.fixture
def corpus(tmp_path):
    archive = tmp_path / "repo.tar"
    entries = {
        "calc.py": b"def add(a, b):\n    return a - b\n",
        "README.md": b"Calculator documentation.",
        ".env": b"PRIVATE_KEY=secret",
        "nested/.env.production": b"PRIVATE_KEY=secret",
        "vendor/library.py": b"vendored code",
        "file.bin": b"binary",
        "nul.py": b"a\0b",
        "bad.py": b"\xff\xfe",
        "large.py": b"x" * 1000,
    }
    with tarfile.open(archive, "w") as tree:
        for path, content in entries.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            tree.addfile(member, io.BytesIO(content))
        link = tarfile.TarInfo("outside.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/outside/secret.py"
        tree.addfile(link)
    scope = MemoryScope("repository", "commit", "issue", "attempt")
    return build_corpus(archive, scope, max_file_bytes=500)


def envelope(data, status=200):
    return httpx.Response(
        status,
        json={"success": True, "data": data, "error": None},
        headers={"X-Request-ID": "request-fixture"},
    )


class FakeHydraAPI:
    def __init__(self):
        self.requests = []
        self.ingested = []
        self.chunks = []
        self.related = {}
        self.statuses = iter(["graph_creation", "completed"])
        self.reject_ingest = False

    def __call__(self, request):
        self.requests.append(request)
        assert request.headers["API-Version"] == "2"
        assert request.headers["Authorization"] == "Bearer private-key"
        if request.url.path == "/databases/status":
            return envelope({"infra": {"ready_for_ingestion": True}})
        if request.url.path == "/context/ingest":
            assert "multipart/form-data" in request.headers["content-type"]
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
                + request.content
            )
            fields = {
                part.get_param("name", header="content-disposition"): part.get_payload(
                    decode=True
                ).decode()
                for part in message.iter_parts()
            }
            assert fields["collection"] == "attempt_attempt"
            assert fields["type"] == "knowledge"
            items = json.loads(fields["app_knowledge"])
            self.ingested.extend(items)
            results = [{"id": item["id"], "status": "accepted"} for item in items]
            return envelope({"results": results[:-1] if self.reject_ingest else results}, 202)
        if request.url.path == "/context/status":
            assert request.url.params["collection"] == "attempt_attempt"
            state = next(self.statuses, "completed")
            return envelope(
                {
                    "statuses": [
                        {"id": source_id, "indexing_status": state}
                        for source_id in request.url.params.get_list("ids")
                    ]
                }
            )
        if request.url.path == "/query":
            return envelope(
                {
                    "chunks": self.chunks,
                    "additional_context": self.related,
                    "graph_context": {"query_paths": [{"secret": "must not inject"}]},
                }
            )
        pytest.fail(f"Unexpected request: {request.method} {request.url.path}")


def memory_for(corpus, api, **kwargs):
    return HydraMemory(
        HydraConfig("database", "private-key"),
        corpus,
        client=httpx.Client(base_url="https://api.hydradb.com", transport=httpx.MockTransport(api)),
        poll_seconds=0,
        **kwargs,
    )


def test_snapshot_manifest_and_exclusions(corpus):
    assert {source.path for source in corpus.sources} == {"calc.py", "README.md"}
    excluded = {item["path"]: item["reason"] for item in corpus.excluded}
    assert excluded[".env"] == "credential_filename"
    assert excluded["nested/.env.production"] == "credential_filename"
    assert excluded["outside.py"] == "non_regular_or_unsafe_path"
    assert excluded["large.py"] == "oversized_file"
    assert excluded["bad.py"] == "non_utf8"
    assert "PRIVATE_KEY" not in json.dumps(corpus.manifest())
    assert all(len(source.sha256) == 64 for source in corpus.sources)


def test_corpus_ids_are_stable_and_attempt_scoped(corpus, tmp_path):
    again = build_corpus(tmp_path / "repo.tar", corpus.scope, max_file_bytes=500)
    assert again.sources == corpus.sources
    other = build_corpus(
        tmp_path / "repo.tar", replace(corpus.scope, attempt_id="other"), max_file_bytes=500
    )
    assert {s.id for s in other.sources}.isdisjoint({s.id for s in corpus.sources})


def test_total_budget_rejects_corpus_before_upload(corpus, tmp_path):
    with pytest.raises(ValueError, match="no data has been uploaded"):
        build_corpus(tmp_path / "repo.tar", corpus.scope, max_total_bytes=1)


def test_prepare_waits_for_completed_graph(corpus):
    api = FakeHydraAPI()
    events = []
    memory = memory_for(corpus, api, report=lambda kind, **data: events.append((kind, data)))
    try:
        result = memory.prepare(timeout=2)
        assert result["status"] == "completed"
        assert memory.ready
        assert len([r for r in api.requests if r.url.path == "/context/status"]) == 2
        assert api.ingested[0]["content"]["text"] == corpus.sources[0].text
        assert all(item["additional_metadata"]["base_commit"] == "commit" for item in api.ingested)
        assert "private-key" not in json.dumps(events)
    finally:
        memory.close()


def test_prepare_provisions_missing_database_before_upload(corpus):
    api = FakeHydraAPI()
    sequence = []
    checks = 0

    def handler(request):
        nonlocal checks
        sequence.append((request.method, request.url.path))
        if request.url.path == "/databases/status":
            checks += 1
            if checks == 1:
                return httpx.Response(404)
            return envelope({"infra": {"ready_for_ingestion": checks >= 3}})
        if request.url.path == "/databases":
            assert json.loads(request.content) == {"database": "database"}
            return envelope({"status": "accepted"}, 202)
        return api(request)

    memory = memory_for(corpus, handler)
    try:
        memory.prepare(timeout=2)
        assert sequence[:5] == [
            ("GET", "/databases/status"),
            ("POST", "/databases"),
            ("GET", "/databases/status"),
            ("GET", "/databases/status"),
            ("POST", "/context/ingest"),
        ]
    finally:
        memory.close()


@pytest.mark.parametrize("failure", ["partial_acceptance", "indexing_failure", "missing_status"])
def test_prepare_never_treats_partial_index_as_ready(corpus, failure):
    api = FakeHydraAPI()
    if failure == "partial_acceptance":
        api.reject_ingest = True
    if failure == "indexing_failure":
        api.statuses = iter(["failed"])

    def handler(request):
        if failure == "missing_status" and request.url.path == "/context/status":
            return envelope({"statuses": []})
        return api(request)

    memory = memory_for(corpus, handler)
    try:
        with pytest.raises(HydraError):
            memory.prepare(timeout=0.03)
        assert not memory.ready
    finally:
        memory.close()


def test_search_scope_freshness_and_related_results(corpus):
    api = FakeHydraAPI()
    first, second = corpus.sources
    api.chunks = [
        {"id": "foreign", "chunk_content": "other repository"},
        {"id": first.id, "chunk_content": "wrong collection", "collection": "foreign"},
        {"id": first.id, "chunk_content": "valid", "chunk_uuid": "one"},
        {"id": first.id, "chunk_content": "valid", "chunk_uuid": "one"},
    ]
    api.related = {"two": {"id": second.id, "chunk_content": "related", "chunk_uuid": "two"}}
    memory = memory_for(corpus, api)
    try:
        memory.prepare(timeout=2)
        hits = memory.search("addition", scope=corpus.scope, limit=8)
        assert [hit["text"] for hit in hits] == ["valid", "related"]
        assert hits[1]["origin"] == "related"
        assert "must not inject" not in json.dumps(hits)
        request = json.loads(api.requests[-1].content)
        assert request["collections"] == [memory.collection]
        assert set(request["ids"]) == set(memory.sources)
        assert request["graph_context"] and request["mode"] == "thinking"
        memory.mark_changed({first.path})
        hits = memory.search("addition", scope=corpus.scope, limit=8)
        assert [hit["path"] for hit in hits] == [second.path]
        assert json.loads(api.requests[-1].content)["ids"] == [second.id]
        before = len(api.requests)
        memory.mark_changed({second.path})
        assert memory.search("addition", scope=corpus.scope, limit=8) == []
        assert len(api.requests) == before
        with pytest.raises(HydraError, match="scope"):
            memory.search("addition", scope=replace(corpus.scope, base_commit="future"), limit=8)
    finally:
        memory.close()


def test_unverifiable_workspace_disables_search(corpus):
    memory = memory_for(corpus, FakeHydraAPI())
    memory.ready = True

    def fail():
        raise RuntimeError("Git unavailable")

    try:
        with pytest.raises(HydraError, match="freshness"):
            memory.refresh(SimpleNamespace(changed_paths=fail))
        with pytest.raises(HydraError, match="not ready"):
            memory.search("task", scope=corpus.scope, limit=8)
    finally:
        memory.close()


def test_request_failure_is_sanitized_and_not_retried(corpus):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(401, text="private-key was invalid")

    memory = memory_for(corpus, handler)
    try:
        with pytest.raises(HydraError, match="HTTP 401") as error:
            memory.prepare(timeout=1)
        assert "private-key" not in str(error.value)
        assert len(requests) == 1
    finally:
        memory.close()


def test_hydra_config_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("HYDRA_DB_DATABASE", "database")
    monkeypatch.setenv("HYDRA_DB_API_KEY", "private-key")
    monkeypatch.delenv("HYDRA_DB_BASE_URL", raising=False)
    assert "private-key" not in repr(HydraConfig.from_env())
    monkeypatch.setenv("HYDRA_DB_BASE_URL", "http://insecure")
    with pytest.raises(ValueError):
        HydraConfig.from_env()


def reuse_manifest(corpus):
    return {
        **corpus.manifest(),
        "status": "completed",
        "database": "database",
        "collection": "attempt_attempt",
        "max_file_bytes": 500,
        "max_total_bytes": 50000000,
    }


def test_reuse_keeps_source_ids_but_uses_new_agent_scope(corpus, tmp_path):
    scope = replace(corpus.scope, attempt_id="new-attempt", instance_id="new-task")
    reused = reuse_corpus(tmp_path / "repo.tar", scope, reuse_manifest(corpus), "database")
    assert reused.scope == scope
    assert reused.sources == corpus.sources


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pending"),
        ("version", 2),
        ("database", "other"),
        ("collection", "attempt_other"),
        ("sources", []),
    ],
)
def test_reuse_rejects_invalid_manifest(corpus, tmp_path, field, value):
    manifest = reuse_manifest(corpus)
    manifest[field] = value
    with pytest.raises(ValueError):
        reuse_corpus(tmp_path / "repo.tar", corpus.scope, manifest, "database")


@pytest.mark.parametrize("field", ["base_commit", "repository"])
def test_reuse_rejects_different_snapshot(corpus, tmp_path, field):
    with pytest.raises(ValueError, match="repository/commit"):
        reuse_corpus(
            tmp_path / "repo.tar",
            replace(corpus.scope, **{field: "other"}),
            reuse_manifest(corpus),
            "database",
        )


def test_reuse_status_then_query_never_ingests(corpus):
    api = FakeHydraAPI()
    api.statuses = iter(["completed"])
    memory = memory_for(corpus, api, existing_collection="attempt_attempt")
    try:
        assert memory.prepare(timeout=2)["reused"]
        memory.mark_changed({corpus.sources[0].path})
        memory.search("test", scope=corpus.scope, limit=8)
        assert [(r.method, r.url.path) for r in api.requests] == [
            ("GET", "/context/status"),
            ("POST", "/query"),
        ]
        query = json.loads(api.requests[-1].content)
        assert query["collections"] == ["attempt_attempt"]
        assert corpus.sources[0].id not in query["ids"]
        with pytest.raises(HydraError, match="forbids"):
            memory._request("POST", "/context/ingest")
        with pytest.raises(HydraError, match="forbids"):
            memory._request("POST", "/databases")
    finally:
        memory.close()


@pytest.mark.parametrize("state", ["graph_creation", "failed", "missing"])
def test_reuse_fails_closed_without_ingestion(corpus, state):
    requests = []

    def handler(request):
        requests.append(request)
        return envelope(
            {
                "statuses": []
                if state == "missing"
                else [{"id": source.id, "indexing_status": state} for source in corpus.sources]
            }
        )

    memory = memory_for(corpus, handler, existing_collection="attempt_attempt")
    try:
        with pytest.raises(HydraError, match="missing or incomplete"):
            memory.prepare(timeout=2)
        assert not memory.ready
        assert [(r.method, r.url.path) for r in requests] == [("GET", "/context/status")]
    finally:
        memory.close()
