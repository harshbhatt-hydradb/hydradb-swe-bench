import ast
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from hydra_agent.agent import Trace
from hydra_agent.bench_data import atomic_json, digest, read_json
from hydra_agent.codewiki_agent import (
    DocumentationAgent,
    UsageBudget,
    citation_audit,
    export_docs,
    generate,
    validate_plan,
)
from hydra_agent.codewiki_data import code_corpus, prepare_record, safe_metadata, validate_rubrics
from hydra_agent.codewiki_eval import evaluate, navigate, validate_judgment
from hydra_agent.codewiki_memory import CodeWikiMemory, index_corpus
from hydra_agent.config import AzureConfig, HydraConfig
from hydra_agent.hydradb import HydraError
from hydra_agent.memory import MemoryScope


@pytest.fixture
def corpus(tmp_path):
    archive = tmp_path / "snapshot.tar"
    files = {
        "src/shop.py": b"def checkout():\n    return 42\n",
        "docs/gold.py": b"SECRET_GOLD",
        "README.md": b"SECRET_REFERENCE",
        "test/fixtures/gold.json": b"SECRET_FIXTURE",
        ".env": b"SECRET_KEY",
        "test/test_shop.py": b"assert checkout() == 42\n",
    }
    with tarfile.open(archive, "w") as tree:
        for name, text in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(text)
            tree.addfile(member, io.BytesIO(text))
        member = tarfile.TarInfo("src/escape.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        tree.addfile(member)
    return code_corpus(
        archive, MemoryScope("https://github.com/test/shop", "a" * 40, "wiki", "fixture")
    )


def test_code_only_corpus_and_metadata_boundary(corpus):
    assert {s.path for s in corpus.sources} == {"src/shop.py", "test/test_shop.py"}
    assert all("SECRET" not in s.text for s in corpus.sources)
    row = {
        "metadata": {
            "repo_name": "Chart.js",
            "repo_url": "https://github.com/chartjs/Chart.js",
            "commit_id": "a" * 40,
        },
        "rubrics": "SECRET_RUBRICS",
        "docs_tree": "SECRET_TREE",
    }
    assert set(safe_metadata(row)) == {"repo_name", "repo_url", "commit_id"}
    row["metadata"]["commit_id"] = "HEAD"
    with pytest.raises(ValueError):
        safe_metadata(row)


def test_agent_cannot_read_reference_or_escape(corpus, tmp_path):
    agent = DocumentationAgent(None, None, corpus, None, None)
    for path in ("../evaluation/reference.json", ".env", "docs/gold.py", "/etc/passwd"):
        with pytest.raises(ValueError):
            agent.tool("read_file", {"path": path, "start_line": 1, "end_line": 10})
    result = agent.tool("read_file", {"path": "src/shop.py", "start_line": 1, "end_line": 10})
    assert result["text"] == "1: def checkout():\n2:     return 42"
    assert result["url"].endswith("/src/shop.py#L1-L2")
    with pytest.raises(ValueError):
        agent.tool("read_file", {"path": "src/shop.py", "start_line": 30, "end_line": 40})


def test_initial_hydra_query_precedes_generation(corpus, tmp_path):
    order = []

    class Memory:
        def search(self, query, **kwargs):
            order.append(("search", query))
            return []

    class Model:
        def complete(self, messages, tools, **kwargs):
            order.append(("model", None))
            assert messages[1]["content"] == "Explain the complete request lifecycle."
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Article"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 20},
            }

    agent = DocumentationAgent(
        Model(),
        Memory(),
        corpus,
        UsageBudget(tmp_path / "usage.json", 100000),
        Trace(tmp_path / "trace.jsonl"),
    )
    assert agent.run("Explain the complete request lifecycle.", label="test") == "Article"
    assert order == [("search", "Explain the complete request lifecycle."), ("model", None)]


def test_failed_retrieval_never_calls_model(corpus, tmp_path):
    class Memory:
        def search(self, *args, **kwargs):
            raise HydraError("unavailable")

    class Model:
        def complete(self, *args, **kwargs):
            pytest.fail("model must not run")

    agent = DocumentationAgent(Model(), Memory(), corpus, None, Trace(tmp_path / "trace.jsonl"))
    with pytest.raises(HydraError):
        agent.run("Explain architecture", label="test")


def test_index_retries_only_failed_sources_and_resumes_without_upload(
    corpus, tmp_path, monkeypatch
):
    from hydra_agent.hydradb import HydraMemory

    monkeypatch.setattr(HydraMemory, "_pause", lambda *a: None)
    uploads = []
    attempts = {s.id: 0 for s in corpus.sources}
    failed_id = corpus.sources[0].id

    def handler(request):
        if request.url.path == "/databases/status":
            data = {"infra": {"ready_for_ingestion": True}}
        elif request.url.path == "/context/ingest":
            from email import policy
            from email.parser import BytesParser

            msg = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
                + request.content
            )
            fields = {
                p.get_param("name", header="content-disposition"): p.get_payload(
                    decode=True
                ).decode()
                for p in msg.iter_parts()
            }
            ids = [i["id"] for i in json.loads(fields["app_knowledge"])]
            uploads.append(ids)
            for sid in ids:
                attempts[sid] += 1
            data = {"results": [{"id": sid, "status": "accepted"} for sid in ids]}
        else:
            data = {
                "statuses": [
                    {
                        "id": sid,
                        "indexing_status": "errored"
                        if sid == failed_id and attempts[sid] == 1
                        else "completed",
                        "error_code": "E6005",
                    }
                    for sid in request.url.params.get_list("ids")
                ]
            }
        return httpx.Response(200, json={"success": True, "data": data})

    config = HydraConfig("database", "test-key")

    def client():
        return httpx.Client(
            base_url="https://api.hydradb.com", transport=httpx.MockTransport(handler)
        )

    path = tmp_path / "index.json"
    memory = index_corpus(config, corpus, path, client=client())
    memory.close()
    assert uploads == [[s.id for s in corpus.sources], [failed_id]]
    assert read_json(path)["status"] == "completed"
    index_corpus(config, corpus, path, client=client()).close()
    assert len(uploads) == 2


def test_partial_index_status_fails_closed(corpus, tmp_path):
    def handler(request):
        if request.url.path == "/databases/status":
            data = {"infra": {"ready_for_ingestion": True}}
        elif request.url.path == "/context/ingest":
            data = {"results": [{"id": s.id} for s in corpus.sources]}
        else:
            data = {"statuses": [{"id": corpus.sources[0].id, "indexing_status": "completed"}]}
        return httpx.Response(200, json={"success": True, "data": data})

    path = tmp_path / "index.json"
    with pytest.raises(HydraError, match="Incomplete"):
        index_corpus(
            HydraConfig("db", "key"),
            corpus,
            path,
            client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
        )
    assert read_json(path)["status"] == "incomplete"


def test_wiki_export_navigation_and_citation_locations(tmp_path, corpus):
    (tmp_path / "pages").mkdir()
    link = f"{corpus.scope.repository}/blob/{corpus.scope.base_commit}/src/shop.py#L1-L2"
    (tmp_path / "pages" / "overview.md").write_text(f"# Overview\n[checkout]({link})\n")
    metadata = {"repo_name": "shop", "commit_id": corpus.scope.base_commit}
    pages = [{"slug": "overview", "title": "Overview", "description": "How checkout works"}]
    docs = export_docs(tmp_path, metadata, pages)
    assert "checkout" in navigate(docs, [["subpages", 0, "content", "markdown"]])[0]["content"]
    assert (
        read_json(tmp_path / "docs_tree.json")["subpages"][0]["content"]["markdown"]
        == "<detail_content>"
    )
    assert citation_audit(docs, corpus)["valid_path_and_line_links"] == 1
    assert "error" in navigate(docs, [["subpages", -1]])[0]
    with pytest.raises(ValueError):
        validate_plan({"pages": [{**pages[0], "slug": "../../escape"}]}, 6)


@pytest.mark.parametrize("score", [True, 0.5, "1", None, -1, 2])
def test_invalid_judge_scores_are_not_passes(score):
    with pytest.raises(ValueError):
        validate_judgment({"score": score, "reasoning": "reason", "evidence": "section"})


def test_official_hierarchical_weights():
    # Execute the same pinned-source extraction logic against an available upstream checkout.
    path = Path("targets/CodeWikiBench/src/judge/judge.py")
    if not path.exists():
        pytest.skip("Optional upstream conformance check; clone pinned CodeWikiBench to run")
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"is_leaf_node", "collect_leaf_requirements", "calculate_scores_bottom_up"}
    ]
    ns = {"json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "scoring_fixture", "exec"), ns)  # noqa: S102 -- upstream scoring conformance
    rubric = [
        {
            "requirements": "Parent",
            "weight": 3,
            "sub_tasks": [{"requirements": "A", "weight": 1}, {"requirements": "B", "weight": 3}],
        },
        {"requirements": "C", "weight": 1},
    ]
    validate_rubrics(rubric)
    leaves = ns["collect_leaf_requirements"](rubric)
    assert [leaf["path"] for leaf in leaves] == ["0.0", "0.1", "1"]
    scored = ns["calculate_scores_bottom_up"](
        rubric, {"0.0": {"score": 1}, "0.1": {"score": 0}, "1": {"score": 1}}
    )
    assert sum(n["weight"] * n["score"] for n in scored) / 4 == 0.4375


def test_judge_error_blocks_overall_score(tmp_path):
    docs = {"title": "test", "content": {"markdown": "Some docs"}, "subpages": []}
    atomic_json(tmp_path / "wiki/structured_docs.json", docs)
    atomic_json(tmp_path / "wiki/docs_tree.json", docs)
    atomic_json(
        tmp_path / "wiki/generation.json", {"status": "completed", "docs_digest": digest(docs)}
    )
    atomic_json(tmp_path / "evaluation/rubrics.json", [{"requirements": "A", "weight": 1}])

    class BrokenModel:
        def __init__(self, config):
            pass

        def complete(self, *args, **kwargs):
            raise TimeoutError("provider timeout")

        def close(self):
            pass

    official = {
        "source_sha256": "test",
        "EVALUATION_SYSTEM_PROMPT": "judge",
        "collect_leaf_requirements": lambda r: [{"requirement": "A", "weight": 1, "path": "0"}],
    }
    events = []
    result = evaluate(
        tmp_path,
        AzureConfig("https://example.com", "test", "key"),
        official,
        workers=1,
        model_factory=BrokenModel,
        report=lambda kind, **data: events.append({"event": kind, **data}),
    )
    assert result["overall_score"] is None
    assert result["status"] == "incomplete"
    assert result["error_paths"] == ["0"]
    assert events[0] == {"event": "evaluation_started", "total": 1, "workers": 1}
    assert events[-1]["completed"] == 1
    assert events[-1]["judged"] == 0
    assert events[-1]["errors"] == 1
    assert events[-1]["error_type"] == "TimeoutError"


def test_generation_resume_and_input_change_detection(tmp_path, corpus):
    calls = []
    events = []

    class Agent:
        def __init__(self):
            self.corpus = corpus
            self.trace = Trace(tmp_path / "trace.jsonl", observer=events.append)

        def run(self, task, *, label, **kwargs):
            calls.append(label)
            if label == "outline":
                return json.dumps(
                    {
                        "pages": [
                            {
                                "slug": "overview",
                                "title": "Overview",
                                "description": "Architecture and public API",
                            }
                        ]
                    }
                )
            return "# Overview\n" + "The checkout function returns 42. " * 15

    root = tmp_path / "wiki"
    metadata = {"repo_name": "shop", "commit_id": "a" * 40}
    generation = generate(Agent(), root, metadata, identity={"model": "fixture"})
    assert generation["status"] == "completed"
    assert calls == ["outline", "overview"]
    assert [event["event"] for event in events] == [
        "outline_ready",
        "page_started",
        "page_completed",
    ]
    events.clear()
    generate(Agent(), root, metadata, identity={"model": "fixture"})
    assert calls == ["outline", "overview"]
    assert events[0] == {"event": "outline_ready", "pages": 1, "resumed": True}
    assert events[1]["event"] == "page_reused"
    with pytest.raises(ValueError, match="settings/code changed"):
        generate(Agent(), root, metadata, identity={"model": "changed"})
    (root / "pages/overview.md").write_text("tampered")
    with pytest.raises(ValueError, match="page changed"):
        generate(Agent(), root, metadata, identity={"model": "fixture"})


def test_saved_benchmark_provenance_detects_reference_changes(tmp_path):
    root = tmp_path / "Chart.js"
    reference = {
        "metadata": {
            "repo_name": "Chart.js",
            "repo_url": "https://github.com/chartjs/Chart.js",
            "commit_id": "a" * 40,
        },
        "rubrics": {"rubrics": [{"requirements": "Gold criterion", "weight": 1}]},
        "docs_tree": {},
        "structured_docs": {},
    }
    atomic_json(root / "evaluation/reference.json", reference)
    (root / "inference").mkdir()
    (root / "inference/snapshot.tar").write_bytes(b"preexisting archive")
    prepare_record(root, "Chart.js", tmp_path / "cache")
    assert set(read_json(root / "inference/task.json")) == {"repo_name", "repo_url", "commit_id"}
    reference["rubrics"]["rubrics"][0]["requirements"] = "tampered"
    atomic_json(root / "evaluation/reference.json", reference)
    with pytest.raises(ValueError, match="provenance"):
        prepare_record(root, "Chart.js", tmp_path / "cache")


def test_large_corpus_queries_keep_all_source_filters(corpus):
    from dataclasses import replace

    from hydra_agent.indexing import Corpus

    sources = [replace(corpus.sources[0], id=f"id_{i}", path=f"src/file{i}.py") for i in range(229)]
    large = Corpus(corpus.scope, sources, [])
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["query"] == "How does checkout compute the total?"
        assert 0 < len(payload["ids"]) <= 200
        assert payload["collections"] == ["attempt_fixture"]
        assert payload["graph_context"] is True
        sid = payload["ids"][-1]
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "chunks": [
                        {
                            "id": sid,
                            "chunk_content": "checkout",
                            "relevancy_score": len(requests),
                            "chunk_uuid": sid + "_chunk",
                            "collection": "attempt_fixture",
                        },
                        {
                            "id": "outside",
                            "chunk_content": "must not leak",
                            "relevancy_score": 1000,
                        },
                    ]
                },
            },
        )

    memory = CodeWikiMemory(
        HydraConfig("db", "key"),
        large,
        client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
    )
    memory.ready = True
    hits = memory.search("How does checkout compute the total?", scope=large.scope, limit=2)
    assert len(requests) == 2
    assert {sid for p in requests for sid in p["ids"]} == {s.id for s in sources}
    assert [hit["score"] for hit in hits] == [2, 1]
    assert all(hit["source_id"] != "outside" for hit in hits)
    assert memory.search_calls == 1
    memory.close()
