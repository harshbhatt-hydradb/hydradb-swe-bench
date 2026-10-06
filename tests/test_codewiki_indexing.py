import json
from collections import Counter
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import httpx
import pytest
from test_codewiki import corpus as corpus_fixture

from hydra_agent import codewiki_memory, hydradb
from hydra_agent.bench_data import atomic_json, read_json
from hydra_agent.config import HydraConfig
from hydra_agent.hydradb import HydraError, HydraMemory
from hydra_agent.indexing import Corpus

corpus = corpus_fixture


def larger(corpus, count):
    return Corpus(
        corpus.scope,
        [replace(corpus.sources[0], id=f"id_{i}", path=f"src/file{i}.py") for i in range(count)],
        [],
    )


def upload_ids(request):
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode() + request.content
    )
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in message.iter_parts()
    }
    items = json.loads(fields["app_knowledge"])
    assert len(items) <= 20
    assert all(
        item["database"] == "db" and item["collection"] == "attempt_fixture" for item in items
    )
    return [item["id"] for item in items]


def response(data):
    return httpx.Response(200, json={"success": True, "data": data})


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 16])
def test_concurrent_batches_are_bounded_and_resume_without_upload(
    corpus, tmp_path, monkeypatch, workers
):
    corpus = larger(corpus, workers * 40 + 20)
    lock = Lock()
    barrier = Barrier(workers, timeout=5)
    active, peaks, uploads = Counter(), Counter(), Counter()
    owner = get_ident()
    saved_on = []
    original_save = codewiki_memory.atomic_json

    def save(*args):
        saved_on.append(get_ident())
        original_save(*args)

    monkeypatch.setattr(codewiki_memory, "atomic_json", save)

    def handler(request):
        route = request.url.path
        if route == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        ids = (
            upload_ids(request)
            if route == "/context/ingest"
            else request.url.params.get_list("ids")
        )
        with lock:
            active[route] += 1
            peaks[route] = max(peaks[route], active[route])
        try:
            # Fill two complete worker groups, followed by one extra batch.
            if int(ids[0].removeprefix("id_")) < workers * 40:
                barrier.wait()
            if route == "/context/ingest":
                with lock:
                    uploads.update(ids)
                return response({"results": [{"id": sid} for sid in ids]})
            return response(
                {"statuses": [{"id": sid, "indexing_status": "completed"} for sid in ids]}
            )
        finally:
            with lock:
                active[route] -= 1

    def run():
        client = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
        memory = codewiki_memory.index_corpus(
            HydraConfig("db", "key"),
            corpus,
            tmp_path / "index.json",
            workers=workers,
            client=client,
        )
        assert memory.ready
        memory.close()

    run()
    assert peaks == {"/context/ingest": workers, "/context/status": workers}
    assert uploads == Counter({s.id: 1 for s in corpus.sources})
    assert set(saved_on) == {owner}
    saved_on.clear()
    run()
    assert uploads == Counter({s.id: 1 for s in corpus.sources})
    assert read_json(tmp_path / "index.json")["status"] == "completed"
    # Revalidation saves once per scan, rather than once per batch.
    assert len(saved_on) < 5 and set(saved_on) == {owner}


@pytest.mark.parametrize("slow_route", ["/context/ingest", "/context/status"])
def test_rolling_pool_starts_next_batch_while_an_earlier_request_is_blocked(
    corpus, tmp_path, slow_route
):
    corpus = larger(corpus, 80)
    first_started, third_started = Event(), Event()
    calls = Counter()
    lock = Lock()

    def handler(request):
        route = request.url.path
        if route == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        ids = (
            upload_ids(request)
            if route == "/context/ingest"
            else request.url.params.get_list("ids")
        )
        with lock:
            calls.update((route, sid) for sid in ids)
        if route == slow_route:
            if ids[0] == "id_0":
                first_started.set()
                assert third_started.wait(5), "Idle worker did not dispatch past a slow request"
            elif ids[0] == "id_20":
                assert first_started.wait(5)
            elif ids[0] == "id_40":
                third_started.set()
        if route == "/context/ingest":
            return response({"results": [{"id": sid} for sid in ids]})
        return response({"statuses": [{"id": sid, "indexing_status": "completed"} for sid in ids]})

    memory = codewiki_memory.index_corpus(
        HydraConfig("db", "key"),
        corpus,
        tmp_path / "index.json",
        workers=2,
        client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
    )
    assert memory.ready and third_started.is_set()
    memory.close()
    assert calls == Counter(
        {
            (route, s.id): 1
            for route in ["/context/ingest", "/context/status"]
            for s in corpus.sources
        }
    )


@pytest.mark.parametrize("failure", [HydraError, KeyboardInterrupt])
def test_dispatch_stops_and_inflight_acknowledgments_are_saved_on_failure(failure):
    first_started, release_first = Event(), Event()
    reserved, accepted, requested = [], [], []
    owner = get_ident()

    def before(group):
        assert get_ident() == owner
        reserved.extend(group)

    def request(batch):
        requested.append(batch)
        if batch[0] == 0:
            first_started.set()
            assert release_first.wait(5)
        else:
            assert first_started.wait(5)
        return batch

    def accept(batch):
        assert get_ident() == owner
        if batch[0] == 20:
            release_first.set()
            raise failure("stop dispatch")
        accepted.append(batch)

    with pytest.raises(failure, match="stop dispatch"):
        codewiki_memory._run_batches(list(range(100)), request, accept, workers=2, before=before)
    assert reserved == [list(range(20)), list(range(20, 40))]
    assert sorted(requested) == reserved
    assert accepted == [list(range(20))]


def test_failed_upload_retains_acknowledgments_and_correct_attempt_counts(corpus, tmp_path):
    corpus = larger(corpus, 60)
    lock = Lock()
    uploads = Counter()
    accepted = set()
    failing = [True]
    observed = []
    failed_ids = {f"id_{i}" for i in range(20, 40)}

    def handler(request):
        if request.url.path == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        if request.url.path == "/context/ingest":
            ids = upload_ids(request)
            with lock:
                uploads.update(ids)
            if set(ids) == failed_ids and failing[0]:
                raise httpx.ReadTimeout("delayed upload acknowledgment", request=request)
            with lock:
                accepted.update(ids)
            return response({"results": [{"id": sid} for sid in ids]})
        return response(
            {
                "statuses": [
                    {"id": sid, "indexing_status": "completed" if sid in accepted else "not_found"}
                    for sid in request.url.params.get_list("ids")
                ]
            }
        )

    def run():
        return codewiki_memory.index_corpus(
            HydraConfig("db", "key"),
            corpus,
            tmp_path / "index.json",
            workers=2,
            client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
            report=lambda kind, **data: observed.append({"event": kind, **data}),
        )

    with pytest.raises(httpx.ReadTimeout):
        run()
    saved = read_json(tmp_path / "index.json")
    assert saved["status"] == "incomplete"
    assert set(saved["statuses"]) == accepted
    # Rolling dispatch may start another batch before the failure is observed.
    # Reserve only dispatched sources and retain every acknowledged upload.
    assert saved["attempts"] == dict(uploads)
    assert [e["uploaded"] for e in observed if e["event"] == "codewiki_upload"][-1] == len(accepted)
    failing[0] = False
    run().close()
    assert read_json(tmp_path / "index.json")["status"] == "completed"
    assert uploads == Counter({s.id: 2 if s.id in failed_ids else 1 for s in corpus.sources})


def completed_manifest(corpus, path):
    saved = {
        **corpus.manifest(),
        "status": "completed",
        "database": "db",
        "collection": "attempt_fixture",
        "attempts": {s.id: 1 for s in corpus.sources},
        "statuses": {s.id: {"id": s.id, "indexing_status": "completed"} for s in corpus.sources},
    }
    atomic_json(path, saved)
    return saved


@pytest.mark.parametrize("workers", [1, 8, 16])
def test_final_verification_uses_requested_concurrency_and_never_uploads(corpus, tmp_path, workers):
    corpus = larger(corpus, workers * 20)
    path = tmp_path / "index.json"
    completed_manifest(corpus, path)
    original = path.read_bytes()
    barrier = Barrier(workers, timeout=5)
    seen = Counter()
    lock = Lock()
    events = []
    owner = get_ident()

    def report(kind, **data):
        if kind == "codewiki_index_verified":
            assert get_ident() == owner
        events.append({"event": kind, **data})

    def handler(request):
        assert (request.method, request.url.path) == ("GET", "/context/status")
        assert request.url.params["database"] == "db"
        assert request.url.params["collection"] == "attempt_fixture"
        ids = request.url.params.get_list("ids")
        assert len(ids) == 20
        with lock:
            seen.update(ids)
        barrier.wait()
        return response({"statuses": [{"id": sid, "indexing_status": "completed"} for sid in ids]})

    client = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
    memory = codewiki_memory.open_index(
        HydraConfig("db", "key"), corpus, path, report, workers=workers, client=client
    )
    assert memory.ready and memory.retrieval_only
    assert seen == Counter({s.id: 1 for s in corpus.sources})
    assert path.read_bytes() == original
    verified = [e for e in events if e["event"] == "codewiki_index_verified"]
    assert [e["checked"] for e in verified] == list(range(20, len(corpus.sources) + 1, 20))
    assert all(e["total"] == len(corpus.sources) for e in verified)
    memory.close()
    assert client.is_closed


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign", "graph_creation"])
def test_final_verification_rejects_invalid_status_and_closes_client(corpus, tmp_path, failure):
    path = tmp_path / "index.json"
    completed_manifest(corpus, path)
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        ids = request.url.params.get_list("ids")
        statuses = [{"id": sid, "indexing_status": "completed"} for sid in ids]
        if failure == "missing":
            statuses.pop()
        elif failure == "duplicate":
            statuses[-1] = statuses[0]
        elif failure == "foreign":
            statuses[-1]["id"] = "foreign_id"
        else:
            statuses[-1]["indexing_status"] = "graph_creation"
        return response({"statuses": statuses})

    client = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
    with pytest.raises(HydraError):
        codewiki_memory.open_index(HydraConfig("db", "key"), corpus, path, workers=8, client=client)
    assert client.is_closed and calls == [("GET", "/context/status")]


def test_final_verification_checks_only_eligible_sources(corpus, tmp_path):
    path = tmp_path / "index.json"
    saved = completed_manifest(corpus, path)
    skipped = corpus.sources[0]
    saved["skipped"] = {skipped.id: {"path": skipped.path}}
    atomic_json(path, saved)

    def handler(request):
        assert request.method == "GET"
        ids = request.url.params.get_list("ids")
        assert set(ids) == {s.id for s in corpus.sources if s != skipped}
        return response({"statuses": [{"id": sid, "indexing_status": "completed"} for sid in ids]})

    memory = codewiki_memory.open_index(
        HydraConfig("db", "key"),
        corpus,
        path,
        client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
    )
    assert memory.ready and memory.dirty_paths == {skipped.path}
    memory.close()


@pytest.mark.parametrize("failures", [1, 3])
def test_resume_status_timeout_retains_checkpoint_without_reupload(
    corpus, tmp_path, monkeypatch, failures
):
    corpus = larger(corpus, 60)
    uploads, status_calls = Counter(), Counter()
    events = []
    resuming = False
    failed_batch = tuple(f"id_{i}" for i in range(20, 40))
    manifest_path = tmp_path / "index.json"
    monkeypatch.setattr(hydradb, "time", SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None))

    def handler(request):
        if request.url.path == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        if request.url.path == "/context/ingest":
            ids = upload_ids(request)
            uploads.update(ids)
            return response({"results": [{"id": sid} for sid in ids]})
        ids = tuple(request.url.params.get_list("ids"))
        status_calls[ids] += 1
        if resuming and ids == failed_batch and status_calls[ids] <= failures:
            raise httpx.ReadTimeout("status timed out", request=request)
        state = "queued" if resuming and failures == 3 and ids[0] == "id_0" else "completed"
        return response({"statuses": [{"id": sid, "indexing_status": state} for sid in ids]})

    def run():
        return codewiki_memory.index_corpus(
            HydraConfig("db", "key"),
            corpus,
            manifest_path,
            workers=1,
            client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
            report=lambda kind, **data: events.append({"event": kind, **data}),
        )

    run().close()
    attempts = read_json(manifest_path)["attempts"]
    status_calls.clear()
    events.clear()
    resuming = True
    if failures == 3:
        with pytest.raises(httpx.ReadTimeout):
            run()
        saved = read_json(manifest_path)
        assert saved["status"] == "incomplete" and saved["error"] == "ReadTimeout"
        # Fresh status from a successful batch survives the following batch's failure.
        assert all(saved["statuses"][f"id_{i}"]["indexing_status"] == "queued" for i in range(20))
        assert len(status_calls) == 2
    else:
        memory = run()
        assert memory.ready
        memory.close()
        assert read_json(manifest_path)["status"] == "completed"
        assert len(status_calls) == 3
    assert status_calls[failed_batch] == min(failures + 1, 3)
    assert all(count == 1 for batch, count in status_calls.items() if batch != failed_batch)
    revalidated = [event for event in events if event["event"] == "codewiki_index_revalidated"]
    assert [event["checked"] for event in revalidated] == ([20] if failures == 3 else [20, 40, 60])
    assert revalidated[-1]["completed"] == (0 if failures == 3 else 60)
    assert all(event["total"] == 60 for event in revalidated)
    assert read_json(manifest_path)["attempts"] == attempts
    assert uploads == Counter({source.id: 1 for source in corpus.sources})
    # A later invocation also recovers the checkpoint after retries were exhausted.
    resuming = False
    run().close()
    saved = read_json(manifest_path)
    assert saved["status"] == "completed" and "error" not in saved
    assert saved["attempts"] == attempts and uploads == Counter(attempts)


def test_stalled_index_reports_sources_and_never_assumes_completion(corpus, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(codewiki_memory, "STALL_WARNING_SECONDS", 0)

    def pause(*args):
        raise HydraError("stop fixture polling")

    monkeypatch.setattr(HydraMemory, "_pause", pause)

    def handler(request):
        if request.url.path == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        if request.url.path == "/context/ingest":
            return response({"results": [{"id": sid} for sid in upload_ids(request)]})
        return response(
            {
                "statuses": [
                    {"id": sid, "indexing_status": "queued"}
                    for sid in request.url.params.get_list("ids")
                ]
            }
        )

    with pytest.raises(HydraError, match="stop fixture"):
        codewiki_memory.index_corpus(
            HydraConfig("db", "key"),
            corpus,
            tmp_path / "index.json",
            client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
            report=lambda kind, **data: events.append({"event": kind, **data}),
        )
    stall = next(e for e in events if e["event"] == "codewiki_index_stalled")
    assert stall["remaining"] == len(corpus.sources)
    assert {s["path"] for s in stall["sources"]} == {s.path for s in corpus.sources}
    assert all(s["status"] == "queued" for s in stall["sources"])
    assert read_json(tmp_path / "index.json")["status"] == "incomplete"


def test_extra_attempt_requires_explicit_limit_and_keeps_completed_sources(
    corpus, tmp_path, monkeypatch
):
    uploads = Counter()
    failed_id = corpus.sources[0].id
    completed_uploads = {source.id: 1 for source in corpus.sources if source.id != failed_id}
    monkeypatch.setattr(HydraMemory, "_pause", lambda *a: None)

    def handler(request):
        if request.url.path == "/databases/status":
            return response({"infra": {"ready_for_ingestion": True}})
        if request.url.path == "/context/ingest":
            ids = upload_ids(request)
            uploads.update(ids)
            return response({"results": [{"id": sid} for sid in ids]})
        return response(
            {
                "statuses": [
                    {
                        "id": sid,
                        "indexing_status": "errored"
                        if sid == failed_id and uploads[sid] < 3
                        else "completed",
                        "error_code": "E9001",
                    }
                    for sid in request.url.params.get_list("ids")
                ]
            }
        )

    def run(**kwargs):
        return codewiki_memory.index_corpus(
            HydraConfig("db", "key"),
            corpus,
            tmp_path / "index.json",
            client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
            **kwargs,
        )

    for _ in range(2):
        with pytest.raises(codewiki_memory.IndexRetryExhausted, match="2 upload attempts.*E9001"):
            run()
        assert uploads == {failed_id: 2, **completed_uploads}
    run(max_attempts=3).close()
    assert uploads == {failed_id: 3, **completed_uploads}
    saved = read_json(tmp_path / "index.json")
    assert saved["status"] == "completed"
    assert saved["attempts"] == dict(uploads)
    run(max_attempts=3).close()
    assert uploads == {failed_id: 3, **completed_uploads}
