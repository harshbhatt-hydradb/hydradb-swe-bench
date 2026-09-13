import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hydra_agent import bench_data, bench_environment, bench_runner, bench_worker
from hydra_agent.agent import Limits
from hydra_agent.bench_data import (
    atomic_json,
    digest,
    load_campaign,
    prepare_campaign,
    read_json,
    safe_task,
)
from hydra_agent.environment import CommandResult


@pytest.fixture
def task():
    return {
        "instance_id": "sympy__sympy-20590",
        "repo": "sympy/sympy",
        "base_commit": "a" * 40,
        "version": "1.7",
        "problem_statement": "Why does this expression return the wrong result?",
        "patch": "SECRET_GOLD",
        "test_patch": "SECRET_TEST",
        "FAIL_TO_PASS": ["SECRET_LABEL"],
        "PASS_TO_PASS": ["SECRET_PASS"],
        "hints_text": "SECRET_HINT",
        "created_at": "SECRET_METADATA",
    }


@pytest.fixture
def campaign(tmp_path, task):
    root = tmp_path / "campaign"
    config = prepare_campaign(
        root,
        [task],
        ids=[task["instance_id"]],
        dataset_revision="b" * 40,
        arms=["baseline", "hydradb"],
        seed=42,
        arch="arm64",
        deployment="test-model",
        endpoint="https://example.invalid/openai/v1",
        reasoning=None,
        limits=Limits(),
    )
    return root, config


def seal(root, config, arm, patch="diff", status="generated"):
    iid = config["ids"][0]
    output = bench_runner.attempt_path(root, iid, arm)
    prediction = {
        "instance_id": iid,
        "model_name_or_path": "hydra-" + arm + "/test-model",
        "model_patch": patch,
    }
    atomic_json(output / "prediction.jsonl", prediction)
    atomic_json(output / "state.json", {"status": status, "prediction_digest": digest(prediction)})
    return output


def test_gold_never_enters_inference_manifest(campaign):
    root, config = campaign
    _, tasks = load_campaign(root)
    assert "SECRET" not in json.dumps(tasks)
    assert "SECRET_GOLD" in (root / "evaluation/dataset.json").read_text()
    assert config["retrieval_policy"] == "required_full_task_first_v1"
    assert len(config["schedule"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo", "../.."),
        ("repo", "a/b/c"),
        ("instance_id", "../bad"),
        ("base_commit", "main"),
        ("environment_setup_commit", 123),
    ],
)
def test_unsafe_task_rejected(task, field, value):
    task[field] = value
    with pytest.raises(ValueError):
        safe_task(task)


@pytest.mark.parametrize("target", ["campaign.json", "inference/tasks.json"])
def test_frozen_records_reject_tampering(campaign, target):
    root, _ = campaign
    path = root / target
    value = read_json(path)
    if isinstance(value, dict):
        value["seed"] += 1
    else:
        value[0]["problem_statement"] = "changed"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="changed"):
        load_campaign(root)


def test_code_change_requires_new_campaign(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(bench_data, "code_digest", lambda: "changed")
    with pytest.raises(ValueError, match="prepare a new campaign"):
        load_campaign(root)
    load_campaign(root, check_code=False)


def test_schedule_reproducible(campaign, tmp_path, task):
    _, config = campaign
    other = prepare_campaign(
        tmp_path / "other",
        [task],
        ids=config["ids"],
        dataset_revision=config["dataset_revision"],
        arms=config["arms"],
        seed=42,
        arch="arm64",
        deployment="test-model",
        endpoint=config["endpoint"],
        reasoning=None,
        limits=Limits(),
    )
    assert config == other


def test_report_unstarted_is_preliminary(campaign):
    root, config = campaign
    result = bench_runner.report(root, config)
    assert result["preliminary"]
    assert result["paired"]["mcnemar_exact_p"] is None
    assert result["arms"]["hydradb"]["assigned"] == 1
    assert result["arms"]["hydradb"]["unknown_usage_attempts"] == 1


def test_evaluation_requires_all_inference_sealed(campaign):
    root, config = campaign
    with pytest.raises(ValueError, match="seal every"):
        bench_runner.evaluate(root, config)
    bench_runner.evaluate(root, config, dry_run=True)


def test_prediction_tampering_rejected(campaign):
    root, config = campaign
    output = seal(root, config, "baseline")
    pred = read_json(output / "prediction.jsonl")
    pred["model_patch"] = "different"
    atomic_json(output / "prediction.jsonl", pred)
    with pytest.raises(ValueError, match="Prediction changed"):
        bench_runner.report(root, config)


def test_report_uses_official_grade_not_summary(campaign):
    root, config = campaign
    seal(root, config, "baseline", patch="")
    output = seal(root, config, "hydradb")
    atomic_json(
        output / "result.json", {"summary": "I failed", "total_tokens": 123, "status": "submitted"}
    )
    path = bench_runner.aggregate(root, config)["hydradb"]
    command, run_id = bench_runner.evaluation_command(root, config, "hydradb", path)
    assert command[command.index("--namespace") + 1] == ""
    assert command[command.index("--max_workers") + 1] == "1"
    directory = root / "evaluation/hydradb"
    atomic_json(directory / "evaluation.json", {"run_id": run_id, "status": "finished"})
    iid = config["ids"][0]
    atomic_json(
        directory
        / "logs/run_evaluation"
        / run_id
        / "hydra-hydradb__test-model"
        / iid
        / "report.json",
        {iid: {"resolved": True}},
    )
    result = bench_runner.report(root, config)
    assert not result["preliminary"]
    assert result["arms"]["hydradb"]["resolved"] == 1
    assert result["arms"]["baseline"]["resolved"] == 0
    assert result["arms"]["hydradb"]["reported_model_tokens"] == 123
    assert result["paired"]["hydra_only"] == 1


def test_failed_assignments_are_retained_and_not_retried(campaign, monkeypatch, tmp_path):
    root, config = campaign
    _, tasks = load_campaign(root)
    calls = []
    monkeypatch.setattr(bench_runner.subprocess, "run", lambda *a, **kw: calls.append(a))
    bench_runner.run_campaign(root, config, tasks, tmp_path / ".env")
    assert len(calls) == 2  # cleanup even when environment is missing
    bench_runner.run_campaign(root, config, tasks, tmp_path / ".env")
    assert len(calls) == 2  # no second attempt
    result = bench_runner.report(root, config)
    assert {r["attempt_status"] for r in result["rows"]} == {"setup_error"}
    assert result["arms"]["baseline"]["assigned"] == 1


def test_cleanup_failure_does_not_lose_state(campaign, monkeypatch, tmp_path):
    root, config = campaign

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 30)

    monkeypatch.setattr(bench_runner.subprocess, "run", fail)
    _, tasks = load_campaign(root)
    bench_runner.run_campaign(root, config, tasks, tmp_path / ".env")
    state = read_json(bench_runner.attempt_path(root, config["ids"][0], "baseline") / "state.json")
    assert state["status"] == "setup_error"
    assert state["cleanup_error"] == "TimeoutExpired"


def test_unsealed_attempt_requires_inspection(campaign, tmp_path):
    root, config = campaign
    first = config["schedule"][0]
    output = bench_runner.attempt_path(root, first["instance_id"], first["arm"])
    atomic_json(output / "state.json", {"status": "running"})
    _, tasks = load_campaign(root)
    with pytest.raises(ValueError, match="Unsealed"):
        bench_runner.run_campaign(root, config, tasks, tmp_path / ".env")


def test_low_disk_preflight_never_calls_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench_environment.shutil, "disk_usage", lambda p: SimpleNamespace(free=1024)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Docker must not run when host disk is insufficient")

    monkeypatch.setattr(bench_environment.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="host disk"):
        bench_environment.preflight(tmp_path, "arm64")


def test_evaluator_env_resolves_context_and_strips_service_keys(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HYDRADB_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DOCKER_CONTEXT", "hydra-swe")
    monkeypatch.setattr(
        bench_environment.subprocess, "check_output", lambda *a, **kw: "unix:///safe/docker.sock\n"
    )
    env = bench_environment.docker_env()
    assert env["DOCKER_HOST"] == "unix:///safe/docker.sock"
    assert not any(k.startswith(("AZURE_", "HYDRA_", "HYDRADB_", "OPENAI_")) for k in env)
    assert "DOCKER_CONTEXT" not in env


def test_worker_rejects_gold_before_loading_credentials(task):
    with pytest.raises(ValueError, match="forbidden"):
        bench_worker.run_job({"task": task, "arm": "hydradb"})


def test_worker_timeout_retains_assignment(campaign, monkeypatch, tmp_path):
    root, config = campaign
    iid = config["ids"][0]
    atomic_json(
        root / "build" / iid / "environment.json",
        {
            "instance_id": iid,
            "base_commit": "a" * 40,
            "image": "sha256:test",
            "repository": "/fixture/repo",
        },
    )
    jobs = []

    def worker(root, command, **kwargs):
        job = read_json(Path(command[command.index("--job") + 1]))
        jobs.append(job)
        assert "SECRET" not in json.dumps(job)
        return CommandResult("deadline", 124, timed_out=True)

    monkeypatch.setattr(bench_runner, "tracked_execute", worker)
    monkeypatch.setattr(bench_runner.subprocess, "run", lambda *a, **kw: None)
    _, tasks = load_campaign(root)
    bench_runner.run_campaign(root, config, tasks, tmp_path / ".env")
    assert len(jobs) == 2
    result = bench_runner.report(root, config)
    assert {row["attempt_status"] for row in result["rows"]} == {"timed_out"}


@pytest.mark.parametrize("arm", ["baseline", "hydradb"])
def test_worker_calls_real_loop_with_full_issue_before_model(campaign, task, monkeypatch, arm):
    from test_harness import ScriptedModel, response

    root, config = campaign
    output = root / "fixture-worker"
    output.mkdir()
    events = []

    class FixtureWorkspace:
        archive = Path("/fixture/base.tar")

        def __init__(self, *args, **kwargs):
            assert kwargs["seed_from_image"]
            assert kwargs["revision"] == task["base_commit"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def patch(self):
            return ""

    class FixtureMemory:
        collection = "fresh-attempt"

        def __init__(self, *args, **kwargs):
            events.append("memory")

        def prepare(self, **kwargs):
            events.append("ingest")
            return {"status": "completed"}

        def search(self, query, **kwargs):
            assert query == task["problem_statement"]
            events.append("full_issue_search")
            return [{"path": "source.py", "text": "source evidence"}]

        initial_search = search

        def close(self):
            pass

    class FixtureModel(ScriptedModel):
        def complete(self, messages, tools, **kwargs):
            events.append("model")
            assert "SECRET" not in json.dumps(messages)
            return super().complete(messages, tools, **kwargs)

        def close(self):
            pass

    azure = SimpleNamespace(
        endpoint=config["endpoint"],
        deployment=config["deployment"],
        reasoning_effort=None,
        api_key="fixture-key",
    )
    monkeypatch.setattr(bench_worker.AzureConfig, "from_env", lambda: azure)
    monkeypatch.setattr(
        bench_worker.HydraConfig,
        "from_env",
        lambda: SimpleNamespace(api_key="fixture-hydra", database="fixture"),
    )
    monkeypatch.setattr(bench_worker, "Workspace", FixtureWorkspace)
    monkeypatch.setattr(bench_worker, "HydraMemory", FixtureMemory)
    monkeypatch.setattr(
        bench_worker, "build_corpus", lambda *a, **kw: SimpleNamespace(manifest=dict)
    )
    monkeypatch.setattr(
        bench_worker,
        "AzureModel",
        lambda conf: FixtureModel(response("finish", {"summary": "Done"})),
    )
    job = {
        "task": safe_task(task),
        "arm": arm,
        "output": str(output),
        "repository": "/fixture/repo",
        "image": "fixture",
        "container": "fixture",
        "attempt_id": "fixture",
        **{
            k: config[k]
            for k in ("endpoint", "deployment", "reasoning_effort", "limits", "index_limits")
        },
    }
    bench_worker.run_job(job)
    assert events == (
        ["memory", "ingest", "full_issue_search", "model"] if arm == "hydradb" else ["model"]
    )
    assert read_json(output / "prediction.jsonl")["instance_id"] == task["instance_id"]


def test_official_recipe_uses_only_empty_grader_placeholders(task, monkeypatch):
    pytest.importorskip("swebench")
    import swebench.harness.test_spec.test_spec as official
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    captured = []

    def make(row):
        captured.append(row)
        return SimpleNamespace(language="py", arch="arm64")

    monkeypatch.setattr(official, "make_test_spec", make)
    monkeypatch.setitem(
        MAP_REPO_VERSION_TO_SPECS["sympy/sympy"]["1.7"], "pre_install", ["mkdir -p /testbed/build"]
    )
    _, install = bench_environment.recipe(task, "arm64")
    assert "SECRET" not in json.dumps(captured)
    assert captured[0]["test_patch"] == ""
    assert "mkdir -p /workspace/build" in install
