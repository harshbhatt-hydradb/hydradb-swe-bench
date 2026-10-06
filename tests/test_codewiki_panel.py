import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hydra_agent.bench_data import atomic_json, digest, read_json

spec = importlib.util.spec_from_file_location(
    "paper_panel", Path(__file__).resolve().parents[1] / "scripts/run_codewiki_paper_panel.py"
)
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


@pytest.fixture
def wiki(tmp_path):
    repository = {
        "repo_name": "svelte",
        "repo_url": "https://github.com/sveltejs/svelte",
        "commit_id": "a" * 40,
    }
    docs = {
        "title": "svelte",
        "description": "Generated docs",
        "content": {},
        "metadata": {"commit": repository["commit_id"]},
        "subpages": [
            {
                "title": "New page",
                "description": "New description",
                "content": {"markdown": "New documentation"},
                "subpages": [],
            }
        ],
    }
    tree = {k: v for k, v in docs.items() if k != "metadata"}
    tree["subpages"] = [{**docs["subpages"][0], "content": {"markdown": "<detail_content>"}}]
    rubric = [{"requirements": "Explain feature", "weight": 1}]
    atomic_json(tmp_path / "wiki/structured_docs.json", docs)
    atomic_json(tmp_path / "wiki/docs_tree.json", tree)
    atomic_json(
        tmp_path / "wiki/generation.json", {"status": "completed", "docs_digest": digest(docs)}
    )
    atomic_json(tmp_path / "evaluation/rubrics.json", {"rubrics": rubric})
    atomic_json(tmp_path / "inference/task.json", repository)
    return tmp_path, {"repository": repository}, rubric


def test_selected_repository_rubric_replaces_the_svelte_rubric():
    module = {}
    exec(  # noqa: S102 -- the paper runner resolves rubrics through its module globals
        "def rubrics():\n return ['svelte']\ndef audit():\n return rubrics()\n",
        module,
    )
    panel.use_repository_rubric(module, ["json"])
    assert module["audit"]() == ["json"]


def test_json_report_compares_only_this_repository():
    section = panel.json_baseline(
        "json",
        {
            "score_percent": 92.13664021164021,
            "propagated_std_percent": 3.468397243175289,
            "leaves_unanimous": 48,
        },
        57,
    )
    assert "92.14 ± 3.47" in section and "48 / 57" in section
    assert "66.06 ± 3.08" in section and "61.28 ± 2.35" in section
    assert "+26.1" in section and "+30.9" in section
    assert (
        panel.json_baseline("svelte", {"score_percent": 80, "propagated_std_percent": 1}, 96) == ""
    )


def test_reasoning_channel_recovers_a_tool_call_without_inventing_a_score():
    from hydra_agent.codewiki_judge_reply import promote_reasoning_channel

    class Message:
        def __init__(self, reasoning):
            self.content = None
            self.tool_calls = None
            self.reasoning = reasoning

    tool = Message(
        'Need Empty.\n{"paths": [["subpages", 1, "content", "markdown"]], "query": "Empty"}'
    )
    assert promote_reasoning_channel(tool)
    assert tool.tool_calls[0].function.name == "docs_navigator"
    assert json.loads(tool.tool_calls[0].function.arguments)["query"] == "Empty"
    prose = Message("The page mentions NotNull, so this looks documented.")
    assert not promote_reasoning_channel(prose)
    assert prose.content is None


def test_wrapper_only_change_keeps_saved_judgments():
    saved = {"inputs": {"wiki": "a"}, "wrapper": "old", "adapter": "same"}
    current = {**saved, "wrapper": "new"}
    assert panel.resume_identity(saved, current) == current
    assert panel.resume_identity(saved, {**current, "adapter": "edited"})["adapter"] == "edited"
    with pytest.raises(ValueError, match="Wiki or evaluator changed"):
        panel.resume_identity(saved, {**current, "inputs": {"wiki": "changed"}})


def test_repository_rubric_reaches_audit_when_runner_is_a_copy():
    module = {}
    exec(  # noqa: S102 -- reproduce runpy returning a shallow copy of the runner
        "def rubrics():\n return ['svelte']\ndef audit():\n return rubrics()\n",
        module,
    )
    copied = dict(module)
    panel.use_repository_rubric(copied, ["json"])
    assert copied["audit"]() == ["json"]
    assert copied["rubrics"]() == ["json"]


def test_other_repository_uses_its_own_pinned_rubric(wiki):
    repo, _protocol, svelte_rubric = wiki
    task = read_json(repo / "inference/task.json")
    task["repo_name"] = "json"
    atomic_json(repo / "inference/task.json", task)
    atomic_json(
        repo / "evaluation/rubrics.json", [{"requirements": "Explain the parser", "weight": 1}]
    )
    loaded, _tree = panel.load_wiki(
        repo, {"repo_name": "json"}, svelte_rubric, strict_protocol=False
    )
    assert loaded["metadata"]["commit"] == task["commit_id"]
    with pytest.raises(ValueError, match="does not match"):
        panel.load_wiki(repo, {"repo_name": "svelte"}, svelte_rubric, strict_protocol=False)


def test_panel_requires_fresh_complete_wiki_with_same_commit_and_rubrics(wiki):
    repo, protocol, rubric = wiki
    docs, tree = panel.load_wiki(repo, protocol, rubric)
    assert docs["subpages"][0]["content"]["markdown"] == "New documentation"
    assert "<detail_content>" in json.dumps(tree)
    atomic_json(
        repo / "wiki/generation.json", {"status": "incomplete", "docs_digest": digest(docs)}
    )
    with pytest.raises(ValueError, match="incomplete"):
        panel.load_wiki(repo, protocol, rubric)
    atomic_json(repo / "wiki/generation.json", {"status": "completed", "docs_digest": digest(docs)})
    with pytest.raises(ValueError, match="commit or rubric"):
        panel.load_wiki(repo, protocol, [{"requirements": "Changed", "weight": 1}])
    tree["subpages"][0]["description"] = "Changed judge hints"
    atomic_json(repo / "wiki/docs_tree.json", tree)
    with pytest.raises(ValueError, match="tree does not match"):
        panel.load_wiki(repo, protocol, rubric)


def test_panel_selects_all_three_times_96_without_importing_old_scores():
    runner = {
        "rubrics": list,
        "leaves": lambda _: {str(i): {"evaluation": {"score": 1}} for i in range(96)},
    }
    selection = panel.fresh_selection(runner, [{"openrouter_id": name} for name in panel.MODEL_IDS])
    assert set(selection) == set(panel.MODEL_IDS)
    assert sum(len(v) for v in selection.values()) == 288
    assert all(v["rerun"] and "score" not in v for m in selection.values() for v in m.values())


def test_wrapper_forwards_generation_options_and_never_single_judge_flags(tmp_path):
    args = panel.parse_args(
        [
            "--output",
            str(tmp_path),
            "--agent-provider",
            "openrouter",
            "--agent-model",
            "openai/gpt-6-astra",
        ]
    )
    flags = panel.generation_args(args)
    assert flags[flags.index("--agent-model") + 1] == "openai/gpt-6-astra"
    assert flags[flags.index("--repos") + 1] == "svelte"
    assert not any("judge" in flag or "panel" in flag or "protocol" in flag for flag in flags)
    json_args = panel.parse_args(["--output", str(tmp_path / "json"), "--repos", "json"])
    assert json_args.repos == ["json"]
    with pytest.raises(SystemExit):
        panel.parse_args(["--judge-model", "anthropic/claude-opus-5"])
    with pytest.raises(SystemExit):
        panel.parse_args(["--repos", "not-a-repo"])
    with pytest.raises(SystemExit):
        panel.parse_args(["--repos", "json", "svelte"])


def test_wrapper_runs_generation_then_pinned_panel_and_can_resume_only_panel(tmp_path, monkeypatch):
    protocol = {"models": [{"openrouter_id": name} for name in panel.MODEL_IDS]}
    monkeypatch.setattr(panel.runpy, "run_path", lambda path: {"verify": lambda: protocol})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(panel.subprocess, "run", run)
    assert panel.main(["--output", str(tmp_path), "--agent-provider", "openrouter"]) == 0
    assert [c[3] for c in commands[:3]] == ["prepare", "index", "generate"]
    assert commands[3][0:3] == ["uv", "run", "--no-project"]
    assert "tiktoken==0.11.0" in commands[3] and commands[3][-1] == "--panel-worker"
    commands.clear()
    assert panel.main(["evaluate", "--output", str(tmp_path)]) == 0
    assert len(commands) == 1 and "--panel-worker" in commands[0]


def test_wrapper_does_not_judge_when_generation_fails(tmp_path, monkeypatch):
    protocol = {"models": [{"openrouter_id": name} for name in panel.MODEL_IDS]}
    monkeypatch.setattr(panel.runpy, "run_path", lambda path: {"verify": lambda: protocol})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[3] == "generate" else 0)

    monkeypatch.setattr(panel.subprocess, "run", run)
    assert panel.main(["--output", str(tmp_path)]) == 1
    assert len(commands) == 3


def test_panel_inherits_campaign_lock_across_processes_without_unlocking_parent(tmp_path):
    script = """
import runpy, sys
from pathlib import Path
panel = runpy.run_path(sys.argv[1])
with panel['campaign_lock'](Path(sys.argv[2]), int(sys.argv[3])):
    print('shared campaign lock')
"""
    path = tmp_path / ".lock"
    with path.open("a") as parent:
        fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [sys.executable, "-c", script, panel.__file__, str(tmp_path), str(parent.fileno())],
            pass_fds=(parent.fileno(),),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "shared campaign lock" in result.stdout
        # Closing the worker's duplicate must leave the controller protected.
        with path.open("a") as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A remaining lock file does not block a subsequent invocation.
    with panel.campaign_lock(tmp_path):
        pass


def test_panel_refuses_unrelated_inherited_descriptor(tmp_path):
    (tmp_path / ".lock").touch()
    with (
        (tmp_path / "wrong.lock").open("a") as wrong,
        pytest.raises(SystemExit, match="does not match"),
        panel.campaign_lock(tmp_path, wrong.fileno()),
    ):
        pytest.fail("must not enter with a different lock")
    with (
        pytest.raises(SystemExit, match="could not be opened or inherited"),
        panel.campaign_lock(tmp_path, 999999),
    ):
        pytest.fail("must not enter with a closed descriptor")


def test_standalone_panel_refuses_active_campaign_and_releases_its_own_lock(tmp_path, monkeypatch):
    protocol = {"models": [{"openrouter_id": name} for name in panel.MODEL_IDS]}
    monkeypatch.setattr(panel.runpy, "run_path", lambda path: {"verify": lambda: protocol})
    entered = []

    def worker(*args):
        entered.append(True)
        with (tmp_path / ".lock").open("a") as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 0

    monkeypatch.setattr(panel, "panel_worker", worker)
    args = ["evaluate", "--output", str(tmp_path), "--panel-worker"]
    with (tmp_path / ".lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="already active"):
            panel.main(args)
    assert not entered
    assert panel.main(args) == 0 and entered == [True]
    with panel.campaign_lock(tmp_path):
        pass


def test_wrapper_forwards_inherited_lock_to_pinned_environment(tmp_path, monkeypatch):
    protocol = {"models": [{"openrouter_id": name} for name in panel.MODEL_IDS]}
    monkeypatch.setattr(panel.runpy, "run_path", lambda path: {"verify": lambda: protocol})
    calls = []
    with (tmp_path / ".lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def run(command, **kwargs):
            calls.append(command)
            descriptor = int(command[command.index("--campaign-lock-fd") + 1])
            assert kwargs["pass_fds"] == (descriptor,) == (owner.fileno(),)
            assert os.fstat(descriptor).st_ino == (tmp_path / ".lock").stat().st_ino
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(panel.subprocess, "run", run)
        assert (
            panel.main(
                ["evaluate", "--output", str(tmp_path), "--campaign-lock-fd", str(owner.fileno())]
            )
            == 0
        )
    assert len(calls) == 1 and calls[0][-1] == "--panel-worker"


@pytest.mark.parametrize("stage,fd", [("run", "3"), ("evaluate", "-1")])
def test_inherited_lock_option_is_internal_to_evaluation(stage, fd):
    with pytest.raises(SystemExit):
        panel.parse_args([stage, "--campaign-lock-fd", fd])
