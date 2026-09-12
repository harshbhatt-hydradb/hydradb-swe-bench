import copy
import json
import os
import subprocess
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from hydra_agent import cli
from hydra_agent.agent import Limits, Trace, run_agent
from hydra_agent.config import AzureConfig
from hydra_agent.environment import Workspace, execute
from hydra_agent.memory import MemoryScope
from hydra_agent.model import AzureModel


def response(name=None, arguments=None, *, finish_reason="tool_calls", usage=20):
    message = {"role": "assistant", "content": None}
    if name:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
        ]
    return {
        "model": "test-model",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"total_tokens": usage},
    }


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        pass


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (path / ".gitignore").write_text("__pycache__/\n.env\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@localhost",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    return path


def test_repair_loop_patch_applies_and_source_preserved(repo, tmp_path):
    (repo / "untracked-secret.txt").write_text("not part of snapshot")
    (repo / ".env").write_text("API_KEY=local-secret")
    model = ScriptedModel(
        response("shell", {"command": "sed -i.bak 's/a - b/a + b/' calc.py && rm calc.py.bak"}),
        response("shell", {"command": "python3 -c 'from calc import add; assert add(2, 3) == 5'"}),
        response("finish", {"summary": "Fixed addition; assertion passed."}),
    )
    with Workspace(repo, backend="local") as workspace:
        assert not (workspace.path / "untracked-secret.txt").exists()
        assert not (workspace.path / ".env").exists()
        result = run_agent(model, workspace, "Fix add", Limits(), Trace(tmp_path / "trace.jsonl"))
        patch = workspace.patch()
    assert result["status"] == "submitted"
    assert result["total_tokens"] == 60
    observations = [m for m in model.requests[-1] if m["role"] == "tool"]
    assert all(json.loads(m["content"])["exit_code"] == 0 for m in observations)
    assert "return a - b" in (repo / "calc.py").read_text()
    applied = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert "+    return a + b" in patch


def test_patch_includes_additions_deletions_and_agent_commits(repo):
    with Workspace(repo, backend="local") as workspace:
        assert workspace.run("rm calc.py; printf 'new\n' > added.txt", 10).exit_code == 0
        workspace.run("git add -A && git -c user.name=X -c user.email=x@y commit -qm change", 10)
        patch = workspace.patch()
    assert "deleted file" in patch
    assert "new file" in patch
    assert "added.txt" in patch


def test_output_and_timeout_are_bounded():
    result = execute(["bash", "-c", "yes x"], timeout=0.15, limit=100)
    assert result.timed_out
    assert result.exit_code == 124
    assert result.truncated
    assert len(result.output) <= 100


def test_closed_output_still_has_timeout():
    result = execute(["bash", "-c", "exec 1>&- 2>&-; sleep 20"], timeout=0.15)
    assert result.timed_out


@pytest.mark.parametrize(
    "reply,status",
    [
        (response("shell", {"command": "false"}, finish_reason="length"), "model_length"),
        (response(finish_reason="stop"), "model_stopped"),
        (RuntimeError("sensitive message"), "api_error"),
    ],
)
def test_stops_do_not_execute_tools(tmp_path, reply, status):
    workspace = SimpleNamespace(run=lambda *args: pytest.fail("should not execute"))
    result = run_agent(
        ScriptedModel(reply), workspace, "task", Limits(), Trace(tmp_path / "trace.jsonl")
    )
    assert result["status"] == status
    assert "sensitive message" not in (tmp_path / "trace.jsonl").read_text()


def test_malformed_tool_call_recovers(tmp_path):
    model = ScriptedModel(response("shell", "not json"), response("finish", {"summary": "Done"}))
    result = run_agent(model, None, "task", Limits(), Trace(tmp_path / "trace.jsonl"))
    assert result["status"] == "submitted"
    assert "error" in json.loads(model.requests[1][-1]["content"])


@pytest.mark.parametrize(
    "limits,status",
    [
        (Limits(max_total_tokens=1), "token_limit"),
        (Limits(max_context_bytes=1), "context_limit"),
    ],
)
def test_budget_admission_makes_no_api_call(tmp_path, limits, status):
    model = ScriptedModel()
    result = run_agent(model, None, "task", limits, Trace(tmp_path / "trace.jsonl"))
    assert result["status"] == status
    assert model.requests == []


def test_step_limit_and_key_redaction(tmp_path):
    model = ScriptedModel(response("invalid", {"secret": "sample-key"}))
    trace = Trace(tmp_path / "trace.jsonl", secrets=("sample-key",))
    result = run_agent(model, None, "sample-key", Limits(max_steps=1), trace)
    assert result["status"] == "step_limit"
    assert "sample-key" not in trace.path.read_text()


@pytest.fixture
def azure_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("AZURE_OPENAI_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-private-key")


def test_azure_config_validation(azure_env, monkeypatch):
    config = AzureConfig.from_env()
    assert config.endpoint == "https://example.openai.azure.com/openai/v1/"
    assert config.api_key not in repr(config)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", config.endpoint)
    assert AzureConfig.from_env().endpoint == config.endpoint
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://insecure.local")
    with pytest.raises(ValueError, match="HTTPS"):
        AzureConfig.from_env()
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        AzureConfig.from_env()


def test_azure_sdk_request(azure_env, monkeypatch):
    from hydra_agent import model as module

    recorded = {}

    def create(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(model_dump=lambda **kw: response("finish", {"summary": "OK"}))

    def client(**kwargs):
        recorded["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(module, "OpenAI", client)
    monkeypatch.setenv("AZURE_OPENAI_REASONING_EFFORT", "medium")
    model = AzureModel(AzureConfig.from_env())
    model.complete([], [], max_tokens=100, timeout=5)
    assert recorded["model"] == "my-deployment"
    assert recorded["max_completion_tokens"] == 100
    assert recorded["reasoning_effort"] == "medium"
    assert "temperature" not in recorded
    assert recorded["client"]["base_url"].endswith("/openai/v1/")


def test_local_command_environment_has_no_azure_key(repo, azure_env):
    with Workspace(repo, backend="local") as workspace:
        result = workspace.run("env", 5)
        assert "AZURE_OPENAI_API_KEY" not in result.output
        assert "test-private-key" not in result.output


def test_cli_artifacts(repo, tmp_path, azure_env, monkeypatch):
    monkeypatch.setattr(
        cli,
        "AzureModel",
        lambda config: ScriptedModel(
            response("shell", {"command": "printf 'new\n' > added.txt"}),
            response("finish", {"summary": "Added file."}),
        ),
    )
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "run",
            "--repo",
            str(repo),
            "--task",
            "Add a file",
            "--backend",
            "local",
            "--allow-local-shell",
            "--output",
            str(output),
            "--instance-id",
            "fixture__repo-1",
        ],
    )
    assert cli.main() == 0
    (attempt,) = output.iterdir()
    prediction = json.loads((attempt / "prediction.jsonl").read_text())
    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert prediction["instance_id"] == "fixture__repo-1"
    assert "added.txt" in prediction["model_patch"]
    assert json.loads((attempt / "result.json").read_text())["status"] == "submitted"
    assert json.loads((attempt / "manifest.json").read_text())["base_commit"]
    for artifact in attempt.iterdir():
        assert "test-private-key" not in artifact.read_text()


@pytest.mark.parametrize(
    ("target", "revision", "message"),
    [
        ("missing", "HEAD", "--repo must point to an existing directory"),
        ("file", "HEAD", "--repo must point to an existing directory"),
        ("plain", "HEAD", "--repo is not an accessible Git repository"),
        ("empty", "HEAD", "HEAD requires at least one commit"),
        ("repo", "missing-revision", "--revision must resolve to a commit"),
    ],
)
def test_cli_reports_invalid_repository(
    target, revision, message, repo, tmp_path, azure_env, monkeypatch, capsys
):
    path = repo if target == "repo" else tmp_path / target
    if target == "file":
        path.write_text("not a repository")
    elif target in {"plain", "empty"}:
        path.mkdir()
        if target == "empty":
            subprocess.run(["git", "init", "-q", str(path)], check=True)

    def unexpected_model(config):
        pytest.fail("Invalid repository inputs must fail before model initialization")

    monkeypatch.setattr(cli, "AzureModel", unexpected_model)
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "run",
            "--repo",
            str(path),
            "--revision",
            revision,
            "--task",
            "Fix",
            "--backend",
            "local",
            "--allow-local-shell",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "CalledProcessError" not in stderr
    assert "test-private-key" not in stderr
    (attempt,) = output.iterdir()
    assert json.loads((attempt / "result.json").read_text()) == {
        "status": "run_error",
        "error_type": "ValueError",
    }


def test_local_backend_requires_opt_in(repo, azure_env, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["hydra-agent", "run", "--repo", str(repo), "--task", "Fix", "--backend", "local"],
    )
    assert cli.main() == 2


def test_real_sdk_serialization_uses_azure_route(azure_env):
    def handle(request):
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-private-key"
        body = json.loads(request.content)
        assert body["model"] == "my-deployment"
        assert body["max_completion_tokens"] == 100
        assert "tools" not in body  # doctor call should omit an empty tool list
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    config = AzureConfig.from_env()
    model = AzureModel(config)
    model.client.close()
    model.client = OpenAI(
        api_key=config.api_key,
        base_url=config.endpoint,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    try:
        result = model.complete(
            [{"role": "user", "content": "Say OK"}], [], max_tokens=100, timeout=5
        )
        assert result["usage"]["total_tokens"] == 4
    finally:
        model.close()


def test_memory_search_receives_explicit_scope(tmp_path):
    scope = MemoryScope("repo", "base-sha", "instance", "attempt")
    observed = []

    class Memory:
        def search(self, query, *, scope, limit):
            observed.append((query, scope, limit))
            return [{"path": "calc.py", "text": "evidence"}]

    model = ScriptedModel(
        response("memory_search", {"query": "add"}), response("finish", {"summary": "Done"})
    )
    result = run_agent(
        model, None, "task", Limits(), Trace(tmp_path / "trace.jsonl"), memory=Memory(), scope=scope
    )
    assert observed == [("add", scope, 8)]
    assert result["status"] == "submitted"
    assert json.loads(model.requests[1][-1]["content"])["hits"][0]["path"] == "calc.py"


def test_cli_records_inference_failure(repo, tmp_path, azure_env, monkeypatch):
    monkeypatch.setattr(cli, "AzureModel", lambda config: ScriptedModel(RuntimeError("secret")))
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "run",
            "--repo",
            str(repo),
            "--task",
            "Fix",
            "--backend",
            "local",
            "--allow-local-shell",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 2
    (attempt,) = output.iterdir()
    assert json.loads((attempt / "result.json").read_text())["status"] == "api_error"
    assert (attempt / "prediction.jsonl").exists()
    assert "secret" not in (attempt / "trajectory.jsonl").read_text()


def test_hydra_cli_indexes_before_model_and_invalidates_edits(
    repo, tmp_path, azure_env, monkeypatch
):
    from test_hydradb import FakeHydraAPI

    from hydra_agent.hydradb import HydraMemory

    api = FakeHydraAPI()
    monkeypatch.setenv("HYDRA_DB_API_KEY", "private-key")
    monkeypatch.setenv("HYDRA_DB_DATABASE", "database")
    memories = []

    # Normalize only the attempt label expected by the shared fake API fixture.
    def memory_factory(config, corpus, report):
        real = HydraMemory(
            config,
            corpus,
            report=report,
            poll_seconds=0,
            client=httpx.Client(base_url=config.base_url, transport=httpx.MockTransport(api)),
        )
        real.collection = "attempt_attempt"
        memories.append(real)
        return real

    monkeypatch.setattr(cli, "HydraMemory", memory_factory)

    def model_factory(config):
        assert memories[0].ready
        source = next(source for source in memories[0].sources.values() if source.path == "calc.py")
        api.chunks = [{"id": source.id, "chunk_content": source.text}]
        return ScriptedModel(
            response("memory_search", {"query": "addition"}),
            response("shell", {"command": "printf 'def add(a, b): return a + b\n' > calc.py"}),
            response("memory_search", {"query": "addition"}),
            response("finish", {"summary": "Fixed addition"}),
        )

    monkeypatch.setattr(cli, "AzureModel", model_factory)
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "run",
            "--repo",
            str(repo),
            "--task",
            "Fix addition",
            "--backend",
            "local",
            "--allow-local-shell",
            "--memory",
            "hydradb",
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    (attempt,) = output.iterdir()
    index = json.loads((attempt / "index-manifest.json").read_text())
    assert index["status"] == "completed"
    assert index["source_count"] == 1  # .gitignore is outside the upload extension policy
    assert memories[0].dirty_paths == {"calc.py"}
    events = [json.loads(line) for line in (attempt / "trajectory.jsonl").read_text().splitlines()]
    tool_results = [e["result"] for e in events if e["event"] == "tool"]
    assert tool_results[0]["hits"][0]["path"] == "calc.py"
    assert tool_results[2]["hits"] == []
    assert "+def add(a, b): return a + b" in (attempt / "patch.diff").read_text()
    assert "return a - b" in (repo / "calc.py").read_text()


def test_cli_reuses_graph_without_ingesting(repo, tmp_path, azure_env, monkeypatch):
    from test_hydradb import FakeHydraAPI

    from hydra_agent.hydradb import HydraMemory
    from hydra_agent.indexing import build_corpus

    with Workspace(repo, backend="local") as workspace:
        old_scope = MemoryScope(str(repo.resolve()), workspace.base_commit, "old-task", "attempt")
        corpus = build_corpus(workspace.archive, old_scope)
    saved = tmp_path / "index-manifest.json"
    saved.write_text(
        json.dumps(
            {
                **corpus.manifest(),
                "status": "completed",
                "database": "database",
                "collection": "attempt_attempt",
                "max_file_bytes": 500000,
                "max_total_bytes": 50000000,
            }
        )
    )
    api = FakeHydraAPI()
    api.statuses = iter(["completed"])
    source = corpus.sources[0]
    api.chunks = [{"id": source.id, "chunk_content": source.text}]
    monkeypatch.setenv("HYDRA_DB_API_KEY", "private-key")
    monkeypatch.setenv("HYDRA_DB_DATABASE", "database")

    def memory_factory(config, corpus, **kwargs):
        return HydraMemory(
            config,
            corpus,
            **kwargs,
            client=httpx.Client(base_url=config.base_url, transport=httpx.MockTransport(api)),
        )

    monkeypatch.setattr(cli, "HydraMemory", memory_factory)
    monkeypatch.setattr(
        cli,
        "AzureModel",
        lambda config: ScriptedModel(
            response("memory_search", {"query": "addition"}),
            response("finish", {"summary": "Inspected existing graph"}),
        ),
    )
    output = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "run",
            "--repo",
            str(repo),
            "--task",
            "Inspect addition",
            "--backend",
            "local",
            "--allow-local-shell",
            "--memory",
            "hydradb",
            "--reuse-index",
            str(saved),
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    assert [(r.method, r.url.path) for r in api.requests] == [
        ("GET", "/context/status"),
        ("POST", "/query"),
    ]
    (attempt,) = output.iterdir()
    index = json.loads((attempt / "index-manifest.json").read_text())
    assert index["reused"] is True
    assert index["scope"] == json.loads(saved.read_text())["scope"]
    assert index["sources"] == corpus.manifest()["sources"]
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["hydradb"]["retrieval_only"] is True
    assert manifest["hydradb"]["collection"] == "attempt_attempt"
    assert manifest["attempt_id"] != old_scope.attempt_id
