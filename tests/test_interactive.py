import json
import os

import pytest
from test_harness import ScriptedModel, response
from test_harness import azure_env as configured_azure  # noqa: F401
from test_harness import repo as source_repo  # noqa: F401

from hydra_agent import cli
from hydra_agent.agent import Limits, Trace
from hydra_agent.environment import Workspace
from hydra_agent.interactive import chat_session, close_pending_tools, terminal_text


@pytest.fixture
def repo(request):
    return request.getfixturevalue("source_repo")


@pytest.fixture
def azure_env(request):
    return request.getfixturevalue("configured_azure")


@pytest.mark.parametrize("backend", ["local", "docker"])
def test_chat_retains_history_edits_and_exports(repo, tmp_path, backend):
    if backend == "docker" and os.environ.get("HYDRA_TEST_DOCKER") != "1":
        pytest.skip("Set HYDRA_TEST_DOCKER=1 to test the real Docker runtime")
    output = tmp_path / "session"
    output.mkdir()
    model = ScriptedModel(
        response("shell", {"command": "printf 'def add(a,b): return a+b\n' > calc.py"}),
        response("finish", {"summary": "Fixed addition"}),
        response("shell", {"command": "python3 -c 'from calc import add; assert add(2,3)==5'"}),
        response("finish", {"summary": "Verified previous edit"}),
    )
    messages = iter(["Fix addition", "Now test that change", "/diff", "/status", "/exit"])
    display = []
    with Workspace(repo, backend=backend) as workspace:
        assert (
            chat_session(
                model,
                workspace,
                Limits(),
                Trace(output / "trajectory.jsonl"),
                output,
                read_input=lambda prompt: next(messages),
                write=display.append,
            )
            == 0
        )
        assert "return a+b" in workspace.run("cat calc.py", 5).output
    assert [m["content"] for m in model.requests[2] if m["role"] == "user"] == [
        "Fix addition",
        "Now test that change",
    ]
    assert "return a+b" in (output / "patch.diff").read_text()
    assert "return a - b" in (repo / "calc.py").read_text()
    session = json.loads((output / "session.json").read_text())
    assert len(session["turns"]) == 2
    assert session["total_tokens"] == 80
    assert "Verified previous edit" in "\n".join(display)


def test_clear_retains_workspace_and_session_budget(repo, tmp_path):
    model = ScriptedModel(
        response("finish", {"summary": "first"}), response("finish", {"summary": "second"})
    )
    messages = iter(["first", "/clear", "second", "/exit"])
    with Workspace(repo, backend="local") as workspace:
        chat_session(
            model,
            workspace,
            Limits(),
            Trace(tmp_path / "trace"),
            tmp_path,
            read_input=lambda prompt: next(messages),
            write=lambda text: None,
        )
    assert [m["content"] for m in model.requests[-1] if m["role"] == "user"] == ["second"]
    assert json.loads((tmp_path / "session.json").read_text())["total_tokens"] == 40


def test_pending_tool_batch_is_closed_without_duplicate_results():
    calls = [{"id": "first"}, {"id": "second"}]
    messages = [
        {"role": "assistant", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "first", "content": "ok"},
    ]
    close_pending_tools(messages)
    close_pending_tools(messages)
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["first", "second"]


def test_chat_interrupt_preserves_protocol_and_accounts_usage(repo, tmp_path):
    class InterruptedModel:
        def complete(self, *args, **kwargs):
            raise KeyboardInterrupt

    messages = iter(["Inspect", "/exit"])
    with Workspace(repo, backend="local") as workspace:
        chat_session(
            InterruptedModel(),
            workspace,
            Limits(),
            Trace(tmp_path / "trace"),
            tmp_path,
            read_input=lambda prompt: next(messages),
            write=lambda text: None,
        )
    session = json.loads((tmp_path / "session.json").read_text())
    assert session["turns"][0]["status"] == "interrupted"
    assert session["total_tokens"] > 0  # conservatively reserve interrupted in-flight inference


def test_chat_eof_and_redaction(repo, tmp_path):
    model = ScriptedModel(response("finish", {"summary": "private-secret\x1b[2J"}))

    def eof(prompt):
        raise EOFError

    display = []
    with Workspace(repo, backend="local") as workspace:
        chat_session(
            model,
            workspace,
            Limits(),
            Trace(tmp_path / "trace", ("private-secret",)),
            tmp_path,
            initial_task="Question",
            read_input=eof,
            write=display.append,
        )
    assert "private-secret" not in "\n".join(display)
    assert "\x1b" not in "\n".join(display)
    assert "private-secret" not in (tmp_path / "conversation.json").read_text()
    assert "private-secret" not in (tmp_path / "session.json").read_text()
    assert terminal_text("a\x1b]52;c;payload\x07b") == "ab"


def test_chat_cli_does_not_require_task(repo, tmp_path, azure_env, monkeypatch):
    monkeypatch.setattr(cli, "AzureModel", lambda config: ScriptedModel())

    def session(*args, **kwargs):
        def eof(prompt):
            raise EOFError

        return chat_session(*args, **kwargs, read_input=eof, write=lambda text: None)

    monkeypatch.setattr(cli, "chat_session", session)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydra-agent",
            "chat",
            "--repo",
            str(repo),
            "--backend",
            "local",
            "--allow-local-shell",
            "--output",
            str(tmp_path / "runs"),
        ],
    )
    assert cli.main() == 0
    (attempt,) = (tmp_path / "runs").iterdir()
    assert (attempt / "session.json").exists()
    assert not (attempt / "prediction.jsonl").exists()  # chat is not a benchmark prediction


def test_docker_interrupt_keeps_workspace(repo):
    if os.environ.get("HYDRA_TEST_DOCKER") != "1":
        pytest.skip("Set HYDRA_TEST_DOCKER=1 to test the real Docker runtime")
    with Workspace(repo) as workspace:
        pid = workspace.run("sleep 60 >/dev/null 2>&1 & echo $!", 5).output.strip()
        assert pid.isdigit()
        workspace.interrupt()
        # A killed child may remain a zombie until PID 1 reaps it, but must not run.
        probe = (
            f"from pathlib import Path; p=Path('/proc/{pid}/stat'); "
            "assert not p.exists() or p.read_text().split()[2] == 'Z'"
        )
        import shlex

        assert workspace.run("python -c " + shlex.quote(probe), 5).exit_code == 0
        assert "return a - b" in workspace.run("cat calc.py", 5).output
