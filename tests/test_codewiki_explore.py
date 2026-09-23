import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_codewiki import corpus as corpus_fixture

from hydra_agent.agent import Trace
from hydra_agent.bench_data import atomic_json, read_json
from hydra_agent.codewiki import main, parser
from hydra_agent.codewiki_agent import AgentFailure, DocumentationAgent, UsageBudget, generate
from hydra_agent.codewiki_explore import (
    VALIDATION_V1_CODE,
    ExplorationConfig,
    explore,
    module_notes,
    validate_discovery,
)
from hydra_agent.codewiki_memory import CodeWikiMemory
from hydra_agent.config import HydraConfig
from hydra_agent.hydradb import HydraError
from hydra_agent.indexing import Corpus

corpus = corpus_fixture


@pytest.fixture
def graph(corpus):
    sources = []
    for name, dependency in (("a", "c"), ("b", "c"), ("c", "a"), ("d", "c")):
        text = f"from {dependency} import run\ndef {name}():\n    return run()\n"
        sources.append(
            replace(
                corpus.sources[0],
                id=name,
                path=f"src/{name}.py",
                text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                size=len(text),
            )
        )
    return Corpus(corpus.scope, sources, [])


def ref(path):
    return {"path": path, "start_line": 1, "end_line": 2}


def discovery(path, targets):
    origin = path or "src/a.py"
    return {
        "summary": f"Source-backed explanation of {origin}",
        "evidence": [ref(origin)],
        "dependencies": [
            {
                "path": target,
                "symbol": "",
                "relation": "module" if path is None else "imports",
                "question": f"How does {target} implement its exported behavior?",
                "evidence": [ref(target if path is None else origin)],
            }
            for target in targets
        ],
        "open_questions": [],
    }


class FixtureAgent(DocumentationAgent):
    def __init__(self, graph, root, *, interrupt=None, targets=None):
        super().__init__(None, None, graph, None, Trace(root / "trace.jsonl"))
        self.calls = []
        self.interrupt = interrupt
        self.targets = targets or {
            None: ["src/a.py", "src/b.py"],
            "src/a.py": ["src/c.py"],
            "src/b.py": ["src/c.py"],
            "src/c.py": ["src/a.py"],
        }

    def run(self, task, *, label, output_validator=None, **kwargs):
        self.calls.append(label)
        self.session_reads = []
        if label == self.interrupt:
            self.interrupt = None
            raise KeyboardInterrupt
        if label == "outline":
            assert "module_notes" in task and self.exploration
            assert self.tool("module_notes", {"contains": "", "offset": 0})["notes"]
            return json.dumps(
                {"pages": [{"slug": "overview", "title": "Overview", "description": "Modules"}]}
            )
        if not label.startswith("explore:"):
            assert "repository survey" in task
            return "# Architecture\n" + "The modules cooperate through the run function. " * 15
        path = label.removeprefix("explore:")
        path = None if path == "repository" else path
        value = discovery(path, self.targets[path])
        for source in self.sources:
            self.tool("read_file", {"path": source, "start_line": 1, "end_line": 3})
        text = json.dumps(value)
        output_validator(text)
        return text


def test_repository_survey_is_one_session_and_resumes(graph, tmp_path):
    agent = FixtureAgent(graph, tmp_path, interrupt="explore:repository")
    config = ExplorationConfig()
    root = tmp_path / "exploration"
    with pytest.raises(KeyboardInterrupt):
        explore(agent, root, {"corpus": "same"}, config)
    saved = read_json(root / "state.json")["state"]
    assert saved["status"] == "interrupted" and saved["completed_order"] == []
    state = explore(agent, root, {"corpus": "same"}, config)
    assert agent.calls == ["explore:repository", "explore:repository"]
    assert state["status"] == "completed" and len(state["nodes"]) == 1
    assert state["deferred"] == [] and state["frontier"] == []
    assert len(state["edges"]) == 2
    assert {edge["path"] for edge in state["edges"]} == {"src/a.py", "src/b.py"}
    assert all(e["verification"] == "inferred_from_read_source" for e in state["edges"])
    calls = list(agent.calls)
    assert explore(agent, root, {"corpus": "same"}, config) == state
    assert agent.calls == calls
    with pytest.raises(ValueError, match="settings or source changed"):
        explore(agent, root, {"corpus": "changed"}, config)
    saved = read_json(root / "state.json")
    saved["state"]["edges"][0]["question"] = "tampered"
    atomic_json(root / "state.json", saved)
    with pytest.raises(ValueError, match="checkpoint changed"):
        explore(agent, root, {"corpus": "same"}, config)


def test_survey_names_modules_without_walking_them(graph, tmp_path):
    for index, config in enumerate(
        (
            ExplorationConfig(),
            ExplorationConfig(max_depth=1),
            ExplorationConfig(max_modules=1),
        )
    ):
        agent = FixtureAgent(graph, tmp_path)
        state = explore(agent, tmp_path / f"explore-{index}", {}, config)
        assert agent.calls == ["explore:repository"]
        assert state["status"] == "completed" and state["deferred"] == []
        assert state["coverage"]["topics_explored"] == 2


def test_discovery_requires_actual_complete_reads_and_allowed_targets(graph, tmp_path):
    agent = DocumentationAgent(None, None, graph, None, None)
    node = {"path": "src/a.py"}
    value = discovery("src/a.py", ["src/c.py"])
    with pytest.raises(ValueError, match="not fully read"):
        validate_discovery(json.dumps(value), agent, node, ExplorationConfig())
    agent.tool("read_file", {"path": "src/a.py", "start_line": 1, "end_line": 3})
    valid = validate_discovery(json.dumps(value), agent, node, ExplorationConfig())
    assert valid["evidence"][0]["source_sha256"] == graph.sources[0].sha256
    value["dependencies"][0]["path"] = "../evaluation/reference.json"
    with pytest.raises(ValueError, match="outside the allowed"):
        validate_discovery(json.dumps(value), agent, node, ExplorationConfig())
    value = discovery("src/a.py", ["src/c.py"])
    value["dependencies"][0]["symbol"] = "nonexistent_symbol"
    with pytest.raises(ValueError, match="absent"):
        validate_discovery(json.dumps(value), agent, node, ExplorationConfig())
    # Even an advertised line range is insufficient when the delivered text was truncated.
    agent.session_reads[0]["text"] = "1: from c import ru"
    with pytest.raises(ValueError, match="not fully read"):
        validate_discovery(json.dumps(discovery("src/a.py", [])), agent, node, ExplorationConfig())


def test_generation_consumes_notes_and_reuses_explored_modules(graph, tmp_path):
    agent = FixtureAgent(graph, tmp_path)
    root = tmp_path / "wiki"
    metadata = {"repo_name": "fixture", "commit_id": graph.scope.base_commit}
    state = generate(agent, root, metadata, identity={}, exploration_config=ExplorationConfig())
    assert state["status"] == "completed" and state["exploration"]["status"] == "completed"
    assert agent.calls == ["explore:repository", "outline", "overview"]
    calls = list(agent.calls)
    generate(agent, root, metadata, identity={}, exploration_config=ExplorationConfig())
    assert agent.calls == calls
    notes = module_notes(agent.exploration, "repository", 0)
    assert notes["total"] == 1 and notes["notes"][0]["path"] is None
    assert module_notes(agent.exploration, "src/c", 0)["notes"] == []
    assert agent.exploration["coverage"]["cited_source_files"] == 2
    assert agent.exploration["coverage"]["unique_evidence_ranges"] == 2


def test_validation_reports_summary_relations_and_wrong_origin_together(graph, tmp_path):
    agent = DocumentationAgent(None, None, graph, None, None)
    for path in agent.sources:
        agent.tool("read_file", {"path": path, "start_line": 1, "end_line": 3})
    value = discovery("src/a.py", ["src/c.py", "src/d.py"])
    value["summary"] = "x" * 1926
    value["dependencies"][0]["relation"] = "re-exports"
    value["dependencies"][1]["evidence"] = [ref("src/b.py")]
    with pytest.raises(ValueError) as exc:
        validate_discovery(json.dumps(value), agent, {"path": "src/a.py"}, ExplorationConfig())
    feedback = str(exc.value)
    assert "1800 characters" in feedback
    assert (
        "dependencies[0].relation: Use relation module/imports/calls/uses/tests/related" in feedback
    )
    assert (
        "dependencies[1].evidence: A dependency must cite its originating module src/a.py"
        in feedback
    )


def test_validation_fix_resumes_existing_notes_before_writing_pages(graph, tmp_path):
    root = tmp_path / "wiki"
    config = ExplorationConfig()
    old_identity = {"code": VALIDATION_V1_CODE, "model": "same", "steps": 10}
    agent = FixtureAgent(graph, tmp_path, interrupt="outline")
    metadata = {"repo_name": "fixture", "commit_id": graph.scope.base_commit}
    with pytest.raises(KeyboardInterrupt):
        generate(agent, root, metadata, identity=old_identity, exploration_config=config)
    previous = read_json(root / "exploration/state.json")
    new_identity = {**old_identity, "code": "validation-fixed"}
    generate(agent, root, metadata, identity=new_identity, exploration_config=config)
    current = read_json(root / "exploration/state.json")
    assert current["state"]["completed_order"][:1] == previous["state"]["completed_order"]
    root_id = previous["state"]["completed_order"][0]
    assert current["state"]["nodes"][root_id] == previous["state"]["nodes"][root_id]
    assert agent.calls.count("explore:repository") == 1
    assert current["state"]["code_upgrades"][0]["previous_code"] == VALIDATION_V1_CODE
    assert current["state"]["code_upgrades"][0]["retained_notes"] == 1
    backups = list((root / "exploration").glob("state-before-validation-fix-*.json"))
    assert len(backups) == 1 and read_json(backups[0]) == previous
    calls = list(agent.calls)
    generate(agent, root, metadata, identity=new_identity, exploration_config=config)
    assert agent.calls == calls


@pytest.mark.parametrize("changed", ["model", "config", "source", "unknown_code", "outline"])
def test_validation_upgrade_rejects_incompatible_checkpoints(graph, tmp_path, changed):
    root = tmp_path / "exploration"
    identity = {"model": "same", "code": VALIDATION_V1_CODE}
    if changed == "unknown_code":
        identity["code"] = "unrelated-implementation"
    config = ExplorationConfig()
    agent = FixtureAgent(graph, tmp_path, interrupt="explore:repository")
    with pytest.raises(KeyboardInterrupt):
        explore(agent, root, identity, config)
    before = (root / "state.json").read_bytes()
    new_identity = {**identity, "code": "validation-fixed"}
    if changed == "model":
        new_identity["model"] = "different"
    elif changed == "config":
        config = ExplorationConfig(max_depth=4)
    elif changed == "source":
        agent.corpus = Corpus(graph.scope, graph.sources[:-1], [])
    elif changed == "outline":
        atomic_json(tmp_path / "outline.json", {"pages": []})
    with pytest.raises(ValueError, match="settings or source changed"):
        explore(agent, root, new_identity, config)
    assert (root / "state.json").read_bytes() == before
    assert not list(root.glob("state-before-validation-fix-*.json"))


def test_generate_cli_wires_exploration_and_search_accounting(graph, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hydra_agent import codewiki
    from hydra_agent.config import AzureConfig

    atomic_json(
        tmp_path / "svelte/inference/task.json",
        {
            "repo_name": "svelte",
            "commit_id": graph.scope.base_commit,
            "repo_url": graph.scope.repository,
        },
    )
    configured = []
    monkeypatch.setattr(codewiki, "get_corpus", lambda *args: graph)
    monkeypatch.setattr(codewiki, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(HydraConfig, "from_env", lambda: HydraConfig("db", "secret"))
    monkeypatch.setattr(
        codewiki, "openrouter_config", lambda name: AzureConfig("https://example.com", name, "key")
    )
    monkeypatch.setattr(
        codewiki,
        "open_index",
        lambda *args: SimpleNamespace(
            search_calls=0,
            configure_search=lambda path, **kw: configured.append((path, kw)),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(codewiki, "AzureModel", lambda config: SimpleNamespace(close=lambda: None))
    agent = FixtureAgent(graph, tmp_path)
    monkeypatch.setattr(codewiki, "DocumentationAgent", lambda *args, **kwargs: agent)
    assert (
        main(
            [
                "generate",
                "--repos",
                "svelte",
                "--output",
                str(tmp_path),
                "--plain",
                "--agent-provider",
                "openrouter",
            ]
        )
        == 0
    )
    assert configured[0][1] == {"max_queries": 0}
    state = read_json(tmp_path / "svelte/wiki/generation.json")
    assert state["identity"]["max_tokens"] == 0
    assert state["exploration"]["coverage"]["topics_explored"] == 2
    report = read_json(tmp_path / "report.json")
    assert report["repositories"][0]["exploration"]["coverage"]["topics_explored"] == 2


def completion(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"total_tokens": 20},
    }


def call(name, args, cid):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_real_agent_corrects_unread_evidence_without_restarting_session(graph, tmp_path):
    value = discovery("src/a.py", [])
    outputs = iter(
        [
            completion(calls=[call("finish", {"content": json.dumps(value)}, "bad")]),
            completion(
                calls=[
                    call("read_file", {"path": "src/a.py", "start_line": 1, "end_line": 3}, "read")
                ]
            ),
            completion(json.dumps(value)),
        ]
    )
    model_messages = []

    class Model:
        def complete(self, messages, tools, **kwargs):
            model_messages.append(json.loads(json.dumps(messages)))
            return next(outputs)

    class Memory:
        searches = 0

        def search(self, *args, **kwargs):
            self.searches += 1
            return []

    memory = Memory()
    agent = DocumentationAgent(
        Model(),
        memory,
        graph,
        UsageBudget(tmp_path / "usage.json", 500000),
        Trace(tmp_path / "trace.jsonl"),
    )
    validator = lambda text: validate_discovery(
        text, agent, {"path": "src/a.py"}, ExplorationConfig()
    )
    result = agent.run("Inspect the module", label="explore", output_validator=validator)
    assert json.loads(result) == value and memory.searches == 1
    assert (
        model_messages[1][-1]["role"] == "tool" and model_messages[1][-1]["tool_call_id"] == "bad"
    )
    assert "not fully read" in model_messages[1][-1]["content"]


@pytest.mark.parametrize("can_finish", [True, False])
def test_artifact_corrections_use_remaining_steps_and_preserve_evidence_checks(
    graph, tmp_path, can_finish
):
    value = discovery("src/a.py", [])
    invalid = {**value, "summary": "x" * 1801}
    final = value if can_finish else invalid
    # Three rejected artifacts previously exhausted the independent two-retry cap,
    # even though the fifth model step remained available.
    outputs = iter(
        [
            completion(
                calls=[
                    call("read_file", {"path": "src/a.py", "start_line": 1, "end_line": 3}, "read")
                ]
            )
        ]
        + [
            completion(calls=[call("finish", {"content": json.dumps(invalid)}, str(i))])
            for i in range(3)
        ]
        + [completion(calls=[call("finish", {"content": json.dumps(final)}, "last")])]
    )
    messages_seen = []

    class Model:
        def complete(self, messages, tools, **kwargs):
            messages_seen.append(json.loads(json.dumps(messages)))
            return next(outputs)

    class Memory:
        searches = 0

        def search(self, *args, **kwargs):
            self.searches += 1
            return []

    memory = Memory()
    agent = DocumentationAgent(
        Model(),
        memory,
        graph,
        UsageBudget(tmp_path / "usage.json", 500000),
        Trace(tmp_path / "trace.jsonl"),
        max_steps=5,
    )
    validator = lambda text: validate_discovery(
        text, agent, {"path": "src/a.py"}, ExplorationConfig()
    )
    if can_finish:
        assert (
            json.loads(agent.run("Inspect", label="explore", output_validator=validator)) == value
        )
    else:
        with pytest.raises(AgentFailure, match="1800 characters"):
            agent.run("Inspect", label="explore", output_validator=validator)
    assert len(messages_seen) == 5 and memory.searches == 1
    assert "1 model steps remain" in messages_seen[-1][-2]["content"]
    assert agent.budget.data["calls"] == 5


def memory_for(graph, handler):
    memory = CodeWikiMemory(
        HydraConfig("db", "key"),
        graph,
        client=httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler)),
    )
    memory.ready = True
    return memory


def test_query_cache_and_budget_survive_resume_without_widening_scope(graph, tmp_path):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert set(payload["ids"]) == {s.id for s in graph.sources}
        assert payload["collections"] == ["attempt_fixture"]
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "chunks": [
                        {
                            "id": "a",
                            "chunk_uuid": "chunk_a",
                            "chunk_content": "module a",
                            "collection": "attempt_fixture",
                        },
                        {"id": "gold", "chunk_content": "must not leak"},
                    ]
                },
            },
        )

    path = tmp_path / "cache.json"
    memory = memory_for(graph, handler)
    memory.configure_search(path, max_queries=1)
    hits = memory.search("How does module a work?", scope=graph.scope, limit=8)
    assert len(hits) == 1
    memory.close()
    resumed = memory_for(graph, handler)
    resumed.configure_search(path, max_queries=1)
    assert resumed.search("How does module a work?", scope=graph.scope, limit=8) == hits
    assert len(requests) == 1 and resumed.search_calls == 1
    with pytest.raises(HydraError, match="search-call limit"):
        resumed.search("Another question", scope=graph.scope, limit=8)
    resumed.configure_search(path, max_queries=0)
    resumed.search("Another question", scope=graph.scope, limit=8)
    assert resumed.search_calls == 2
    resumed.close()
    changed = memory_for(Corpus(graph.scope, graph.sources[:1], []), handler)
    with pytest.raises(ValueError, match="different sources"):
        changed.configure_search(path)
    changed.close()


def test_failed_search_is_reserved_but_not_cached(graph, tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("fixture timeout", request=request)

    path = tmp_path / "cache.json"
    memory = memory_for(graph, handler)
    memory.configure_search(path, max_queries=1)
    with pytest.raises(httpx.ReadTimeout):
        memory.search("How does module a work?", scope=graph.scope, limit=8)
    assert read_json(path)["queries"] == 1 and read_json(path)["results"] == {}
    resumed = memory_for(graph, handler)
    resumed.configure_search(path, max_queries=1)
    with pytest.raises(HydraError, match="search-call limit"):
        resumed.search("How does module a work?", scope=graph.scope, limit=8)
    memory.close()
    resumed.close()


def test_real_agent_explores_retrieves_notes_and_writes_wiki(graph, tmp_path):
    queries, notes_delivered = [], []
    targets = {
        None: ["src/a.py", "src/b.py"],
        "src/a.py": ["src/c.py"],
        "src/b.py": ["src/c.py"],
        "src/c.py": ["src/a.py"],
    }

    def handler(request):
        payload = json.loads(request.content)
        queries.append(payload["query"])
        return httpx.Response(200, json={"success": True, "data": {"chunks": []}})

    class Model:
        def complete(self, messages, tools, **kwargs):
            task = messages[1]["content"]
            names = {t["function"]["name"] for t in tools}
            assert "module_notes" in names
            reads = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            if task.startswith(("Discover distinct", "Explore source module")):
                path = (
                    None
                    if task.startswith("Discover")
                    else task.split("Explore source module ")[1].split(" and symbol")[0]
                )
                value = discovery(path, targets[path])
                if not any(isinstance(r, dict) and "total_lines" in r for r in reads):
                    paths = {e["path"] for e in value["evidence"]}
                    paths.update(e["path"] for d in value["dependencies"] for e in d["evidence"])
                    calls = [
                        call("read_file", {"path": p, "start_line": 1, "end_line": 3}, str(i))
                        for i, p in enumerate(sorted(paths))
                    ]
                    if path == "src/a.py":
                        calls.append(
                            call(
                                "memory_search",
                                {"query": "How does module c implement run?"},
                                "followup",
                            )
                        )
                    return completion(calls=calls)
                return completion(
                    calls=[call("finish", {"content": json.dumps(value)}, "finished")]
                )
            if not any(isinstance(r, dict) and "notes" in r for r in reads):
                return completion(
                    calls=[call("module_notes", {"contains": "", "offset": 0}, "notes")]
                )
            notes_delivered.append(next(r for r in reads if isinstance(r, dict) and "notes" in r))
            if task.startswith("Plan a comprehensive"):
                return completion(
                    json.dumps(
                        {
                            "pages": [
                                {
                                    "slug": "overview",
                                    "title": "Overview",
                                    "description": "Module interactions",
                                }
                            ]
                        }
                    )
                )
            return completion(
                "# Module interactions\n" + "Modules a and b both import run from c. " * 15
            )

    memory = memory_for(graph, handler)
    memory.configure_search(tmp_path / "retrieval-cache.json", max_queries=0)
    agent = DocumentationAgent(
        Model(),
        memory,
        graph,
        UsageBudget(tmp_path / "usage.json", 1000000),
        Trace(tmp_path / "trace.jsonl"),
    )
    state = generate(
        agent,
        tmp_path / "wiki",
        {"repo_name": "fixture", "commit_id": graph.scope.base_commit},
        identity={"model": "fixture"},
        exploration_config=ExplorationConfig(),
    )
    assert (
        state["status"] == "completed" and state["exploration"]["coverage"]["topics_explored"] == 2
    )
    assert len(queries) == 3
    assert len(notes_delivered) == 2 and all(len(n["notes"]) == 1 for n in notes_delivered)
    calls = len(queries)
    generate(
        agent,
        tmp_path / "wiki",
        {"repo_name": "fixture", "commit_id": graph.scope.base_commit},
        identity={"model": "fixture"},
        exploration_config=ExplorationConfig(),
    )
    assert len(queries) == calls  # Fully resumed generation makes no model or search calls.
    memory.close()


def test_cli_has_no_budget_flags():
    args = parser().parse_args([])
    assert args.agent_provider == "azure"
    assert args.agent_model == "openai/gpt-6-astra"
    with pytest.raises(SystemExit):
        main(["--max-generation-tokens", "0"])
