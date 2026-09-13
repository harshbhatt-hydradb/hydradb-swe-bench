import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_benchmark import campaign as campaign_fixture
from test_benchmark import seal
from test_benchmark import task as task_fixture

from hydra_agent import bench_cli, bench_process, bench_runner
from hydra_agent.bench_data import atomic_json, load_campaign, read_json
from hydra_agent.environment import CommandResult

campaign = campaign_fixture
task = task_fixture


@pytest.fixture
def cli(monkeypatch, task):
    monkeypatch.setattr(bench_cli, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(
        bench_cli.AzureConfig,
        "from_env",
        lambda: SimpleNamespace(
            endpoint="https://example.invalid/openai/v1",
            deployment="test-model",
            reasoning_effort=None,
        ),
    )
    monkeypatch.setattr(bench_cli.HydraConfig, "from_env", lambda: None)
    monkeypatch.setattr(bench_cli, "check_evaluator", lambda: None)
    monkeypatch.setattr(bench_cli, "download_tasks", lambda revision: ("b" * 40, [task]))
    monkeypatch.setattr(bench_cli, "preflight", lambda *a, **kw: {})
    monkeypatch.setattr(bench_cli, "build_task", lambda *a, **kw: None)
    monkeypatch.setattr(bench_process, "identity", lambda pid: f"fixture-{pid}")
    return bench_cli


def test_one_command_prepares_runs_grades_and_resumes(cli, tmp_path, monkeypatch):
    root = tmp_path / "results"
    calls = []

    def run(root, config, tasks, env, *, resume):
        assert resume
        calls.append("run")
        for arm in config["arms"]:
            seal(root, config, arm, patch="")

    def evaluate(root, config, *, resume, dry_run):
        assert resume and not dry_run
        calls.append("evaluate")

    monkeypatch.setattr(cli, "run_campaign", run)
    monkeypatch.setattr(cli, "evaluate", evaluate)
    assert cli.main(["--output", str(root)]) == 0
    config, tasks = load_campaign(root)
    assert config["ids"] == ["sympy__sympy-20590"]
    assert config["arms"] == ["baseline", "hydradb"]
    assert "patch" not in tasks[0]
    monkeypatch.setattr(cli, "download_tasks", lambda *a: pytest.fail("Must reuse saved tasks"))
    assert cli.main(["--output", str(root)]) == 0
    assert calls == ["run", "evaluate", "run", "evaluate"]
    assert load_campaign(root)[0] == config
    assert (root / "report.md").exists()


@pytest.mark.parametrize("change", ["fresh", "code", "limits"])
def test_new_run_is_automatic_and_preserves_previous_results(cli, campaign, monkeypatch, change):
    root, saved = campaign
    (root / "sentinel").write_text("keep this")
    args = cli.parse_args(["--output", str(root), "--arch", "arm64"])
    if change == "fresh":
        args.fresh = True
    elif change == "code":
        monkeypatch.setattr(cli, "code_digest", lambda: "new-code")
    else:
        args.max_steps = 5
    config, _ = cli.ensure_run(root, args)
    archives = list(root.parent.glob(root.name + "-*"))
    assert len(archives) == 1
    assert (archives[0] / "sentinel").read_text() == "keep this"
    assert not (root / "sentinel").exists()
    assert config["ids"] == saved["ids"]
    assert config["config_digest"] != saved["config_digest"]
    if change == "limits":
        assert config["limits"]["max_steps"] == 5


def test_invalid_new_selection_preserves_existing_run(cli, campaign):
    root, saved = campaign
    args = cli.parse_args(["--output", str(root), "--instances", "missing__task"])
    with pytest.raises(ValueError, match="Selected IDs"):
        cli.ensure_run(root, args)
    assert load_campaign(root)[0] == saved


def test_regrade_does_not_require_model_credentials_or_current_code(cli, campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(cli.AzureConfig, "from_env", lambda: pytest.fail("Grading needs no model"))
    monkeypatch.setattr(cli, "ensure_run", lambda *a: pytest.fail("Must use saved predictions"))
    calls = []
    monkeypatch.setattr(cli, "evaluate", lambda *a, **kw: calls.append(kw))
    assert cli.main(["evaluate", "--output", str(root), "--dry-run"]) == 0
    assert calls == [{"dry_run": True, "resume": True}]


def test_resume_retries_interrupted_attempt_and_keeps_generated(campaign, monkeypatch):
    root, config = campaign
    iid = config["ids"][0]
    completed = seal(root, config, "baseline")
    old_prediction = (completed / "prediction.jsonl").read_bytes()
    interrupted = bench_runner.attempt_path(root, iid, "hydradb")
    atomic_json(interrupted / "state.json", {"status": "running"})
    (interrupted / "worker.log").write_text("old attempt")
    atomic_json(root / "evaluation/hydradb/evaluation.json", {"status": "running"})
    atomic_json(
        root / "build" / iid / "environment.json",
        {
            "instance_id": iid,
            "base_commit": "a" * 40,
            "image": "fixture",
            "repository": "/fixture",
        },
    )
    calls = []

    def worker(root, command, **kwargs):
        job = read_json(Path(command[command.index("--job") + 1]))
        calls.append(job["arm"])
        seal(root, config, "hydradb", patch="new prediction")
        return CommandResult("done", 0)

    monkeypatch.setattr(bench_runner, "tracked_execute", worker)
    monkeypatch.setattr(bench_runner.subprocess, "run", lambda *a, **kw: None)
    bench_runner.run_campaign(root, config, load_campaign(root)[1], root / ".env", resume=True)
    assert calls == ["hydradb"]
    assert (completed / "prediction.jsonl").read_bytes() == old_prediction
    assert next((root / "history").rglob("worker.log")).read_text() == "old attempt"
    assert not (root / "evaluation/hydradb").exists()


def test_evaluation_retries_error_and_skips_complete_reports(campaign, monkeypatch):
    root, config = campaign
    seal(root, config, "baseline", patch="")
    seal(root, config, "hydradb")
    predictions = bench_runner.aggregate(root, config)
    _, run_id = bench_runner.evaluation_command(root, config, "hydradb", predictions["hydradb"])
    directory = root / "evaluation/hydradb"
    atomic_json(directory / "evaluation.json", {"run_id": run_id, "status": "running"})
    calls = []

    def evaluator(root, command, **kwargs):
        calls.append(command)
        iid = config["ids"][0]
        atomic_json(
            directory
            / "logs/run_evaluation"
            / run_id
            / "hydra-hydradb__test-model"
            / iid
            / "report.json",
            {iid: {"resolved": False}},
        )
        return CommandResult("graded", 0)

    monkeypatch.setattr(bench_runner, "tracked_execute", evaluator)
    monkeypatch.setattr(bench_runner, "check_evaluator", lambda: None)
    monkeypatch.setattr(bench_runner, "docker_env", dict)
    monkeypatch.setattr(bench_runner.subprocess, "run", lambda *a, **kw: None)
    bench_runner.evaluate(root, config, resume=True)
    assert len(calls) == 2
    bench_runner.evaluate(root, config, resume=True)
    assert len(calls) == 2
    assert list((root / "history").rglob("evaluation.json"))
    # A zero exit without a task report must remain retryable.
    next((directory / "logs").rglob("report.json")).unlink()
    bench_runner.evaluate(root, config, resume=True)
    assert len(calls) == 3


def test_lock_reports_active_owner_and_never_signals_recycled_pid(tmp_path, monkeypatch):
    root = tmp_path / "run"
    monkeypatch.setattr(bench_process, "identity", lambda pid: "current")
    with (
        bench_process.run_lock(root),
        pytest.raises(ValueError, match="--restart"),
        bench_process.run_lock(root),
    ):
        pytest.fail("Must not enter a locked run")
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("Recycled PID must not be signalled"))
    assert not bench_process.signal_record({"pid": 12345, "identity": "old"}, signal.SIGTERM)


@pytest.mark.parametrize("mode", ["restart", "stop", "crash"])
def test_restart_stops_controller_and_its_worker(tmp_path, mode):
    root = tmp_path / "run"
    root.mkdir()
    script = """
import signal, sys
from pathlib import Path
from hydra_agent.bench_process import run_lock, tracked_execute
root = Path(sys.argv[1])
def interrupt(*args):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)
try:
    with run_lock(root):
        tracked_execute(root, [sys.executable, '-c', 'import time; time.sleep(30)'], timeout=40)
except KeyboardInterrupt:
    (root / 'stopped').touch()
"""
    proc = subprocess.Popen([sys.executable, "-c", script, str(root)])
    try:
        deadline = time.monotonic() + 10
        child = root / ".child.json"
        while not child.exists():
            assert proc.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        record = read_json(child)
        assert record["identity"]
        if mode == "crash":
            proc.kill()
            proc.wait(timeout=5)
        with bench_process.run_lock(root, restart=mode == "restart", stop=mode == "stop"):
            if mode != "crash":
                assert (root / "stopped").exists()
            deadline = time.monotonic() + 5
            while bench_process.identity(record["pid"]):
                assert time.monotonic() < deadline
                time.sleep(0.05)
        assert proc.wait(timeout=5) == (-signal.SIGKILL if mode == "crash" else 0)
        assert not child.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        bench_process.recover_child(root)


def test_unrelated_output_is_never_archived(cli, tmp_path):
    root = tmp_path / "unrelated"
    root.mkdir()
    (root / "important.txt").write_text("keep")
    assert cli.main(["--output", str(root), "--fresh"]) == 2
    assert (root / "important.txt").read_text() == "keep"


def test_active_legacy_pipeline_is_not_treated_as_an_orphan(cli, campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(cli, "recover_child", lambda *a: pytest.fail("Legacy worker is active"))
    with (root / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cli.main(["--output", str(root)]) == 2


def test_evaluator_cleanup_targets_only_exact_run_containers(monkeypatch):
    calls = []
    monkeypatch.setattr(bench_runner.subprocess, "run", lambda *a, **kw: calls.append(a[0]))
    bench_runner.cleanup_evaluation_containers({"ids": ["sympy__sympy-20590"]}, "hydra_test", {})
    assert calls == [["docker", "rm", "-f", "sweb.eval.sympy__sympy-20590.hydra_test"]]


def test_launcher_help_forwards_arguments_without_ssd(tmp_path):
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["./benchmark", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"]},
        check=True,
    )
    assert result.stdout.splitlines() == ["run", "--extra", "benchmark", "hydra-bench", "--help"]
