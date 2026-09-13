"""Serial paired campaigns: immutable attempts, official evaluation, explicit failures."""

import json
import math
import subprocess
import sys
import uuid
from pathlib import Path

from .bench_data import atomic_json, digest, read_json
from .bench_environment import check_evaluator, docker_env
from .bench_process import tracked_execute


def attempt_path(root, instance_id, arm):
    return root / "attempts" / arm / instance_id


def archive_artifact(root: Path, path: Path) -> None:
    target = root / "history" / uuid.uuid4().hex / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)


def run_campaign(
    root: Path, config: dict, tasks: list[dict], env_file: Path, *, resume=False
) -> None:
    lookup = {task["instance_id"]: task for task in tasks}
    for assignment in config["schedule"]:
        iid, arm = assignment["instance_id"], assignment["arm"]
        output = attempt_path(root, iid, arm)
        state_path = output / "state.json"
        if output.exists() and resume:
            state = read_json(state_path) if state_path.exists() else {}
            if state.get("status") != "generated":
                container = state.get("container", "")
                if (
                    container.startswith("hydra-bench-")
                    and state.get("attempt_id") == container[12:]
                ):
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                print(f"Retry {arm}/{iid}; previous attempt saved in history/", flush=True)
                archive_artifact(root, output)
                evaluation = root / "evaluation" / arm
                if evaluation.exists():
                    archive_artifact(root, evaluation)
        if output.exists():
            if not state_path.exists():
                raise ValueError(f"Incomplete attempt metadata at {output}; refusing to rerun")
            state = read_json(state_path)
            if state["status"] == "running":
                raise ValueError(
                    f"Unsealed attempt at {output}; inspect its worker/container before recovery"
                )
            print(f"Skip {arm}/{iid}: {state['status']}", flush=True)
            continue
        output.mkdir(parents=True)
        attempt = uuid.uuid4().hex
        container = "hydra-bench-" + attempt
        state = {"status": "running", "attempt_id": attempt, "container": container, **assignment}
        atomic_json(state_path, state)  # claim before inference: never silently sample twice
        try:
            receipt = read_json(root / "build" / iid / "environment.json")
            if (
                receipt["instance_id"] != iid
                or receipt["base_commit"] != lookup[iid]["base_commit"]
            ):
                raise ValueError("Environment receipt does not match task")
            job = {
                "task": lookup[iid],
                "arm": arm,
                "attempt_id": attempt,
                "container": container,
                "output": str(output),
                "image": receipt["image"],
                "repository": receipt["repository"],
                **{
                    k: config[k]
                    for k in (
                        "endpoint",
                        "deployment",
                        "reasoning_effort",
                        "limits",
                        "index_limits",
                    )
                },
            }
            atomic_json(output / "job.json", job)
            print(f"Run {arm}/{iid}", flush=True)
            deadline = config["limits"]["wall_seconds"] + 120
            if arm == "hydradb":
                deadline += config["index_limits"]["timeout"]
            result = tracked_execute(
                root,
                [
                    sys.executable,
                    "-m",
                    "hydra_agent.bench_worker",
                    "--job",
                    str(output / "job.json"),
                    "--env-file",
                    str(env_file),
                ],
                timeout=deadline,
                limit=100000,
            )
            (output / "worker.log").write_text(result.output)
            prediction = output / "prediction.jsonl"
            state["status"] = (
                "timed_out"
                if result.timed_out
                else ("generated" if prediction.exists() else "inference_error")
            )
            state["worker_exit_code"] = result.exit_code
            if prediction.exists():
                state["prediction_digest"] = digest(json.loads(prediction.read_text()))
        except Exception as exc:  # noqa: BLE001 -- preserve every assignment and do not retry model
            state.update(status="setup_error", error_type=type(exc).__name__)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            raise
        finally:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container], capture_output=True, timeout=30, check=False
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                state["cleanup_error"] = type(exc).__name__
            finally:
                atomic_json(state_path, state)
    aggregate(root, config)


def aggregate(root: Path, config: dict) -> dict[str, Path]:
    paths = {}
    for arm in config["arms"]:
        rows = []
        for iid in config["ids"]:
            output = attempt_path(root, iid, arm)
            path = output / "prediction.jsonl"
            state_path = output / "state.json"
            if path.exists() and not state_path.exists():
                raise ValueError("Prediction lacks attempt metadata")
            if path.exists() and state_path.exists():
                row = json.loads(path.read_text())
                state = read_json(state_path)
                if digest(row) != state.get("prediction_digest"):
                    raise ValueError("Prediction changed or attempt was interrupted before sealing")
                if row["instance_id"] != iid or set(row) != {
                    "instance_id",
                    "model_name_or_path",
                    "model_patch",
                }:
                    raise ValueError("Malformed prediction")
                if (
                    not isinstance(row["model_patch"], str)
                    or row["model_name_or_path"] != "hydra-" + arm + "/" + config["deployment"]
                ):
                    raise ValueError("Malformed prediction or unexpected model name")
            else:
                row = {
                    "instance_id": iid,
                    "model_name_or_path": "hydra-" + arm + "/" + config["deployment"],
                    "model_patch": "",
                }
            rows.append(row)
        path = root / "predictions" / (arm + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temporary.replace(path)
        paths[arm] = path
    return paths


def evaluation_command(
    root: Path, config: dict, arm: str, predictions: Path
) -> tuple[list[str], str]:
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    run_id = "hydra_" + digest([config["config_digest"], arm, rows])[:24]
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(root / "evaluation" / "dataset.json"),
        "--split",
        "test",
        "--predictions_path",
        str(predictions),
        "--run_id",
        run_id,
        "--max_workers",
        "1",
        "--timeout",
        "1800",
        "--namespace",
        "",
        "--cache_level",
        "env",
        "--report_dir",
        str(root / "evaluation" / arm),
    ]
    return command, run_id


def cleanup_evaluation_containers(config: dict, run_id: str, env: dict) -> None:
    # Names follow the pinned evaluator's TestSpec.get_instance_container_name.
    names = [f"sweb.eval.{iid.lower()}.{run_id}" for iid in config["ids"]]
    try:
        subprocess.run(
            ["docker", "rm", "-f", *names],
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("Evaluator container cleanup will be retried on the next run.", flush=True)


def recover_containers(root: Path, config: dict) -> None:
    """Reconcile containers after stopping the recorded host process, under both locks."""
    for assignment in config["schedule"]:
        path = attempt_path(root, assignment["instance_id"], assignment["arm"]) / "state.json"
        if not path.exists():
            continue
        state = read_json(path)
        container = state.get("container", "")
        if (
            state.get("status") == "running"
            and state.get("attempt_id")
            and container == "hydra-bench-" + state["attempt_id"]
        ):
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, timeout=30, check=False
            )
    for arm in config["arms"]:
        path = root / "evaluation" / arm / "evaluation.json"
        if path.exists():
            receipt = read_json(path)
            if receipt["status"] == "running":
                cleanup_evaluation_containers(config, receipt["run_id"], docker_env())


def evaluate(root: Path, config: dict, *, dry_run=False, resume=False) -> None:
    if not dry_run:
        for assignment in config["schedule"]:
            state = attempt_path(root, assignment["instance_id"], assignment["arm"]) / "state.json"
            if not state.exists() or read_json(state)["status"] == "running":
                raise ValueError("Finish and seal every inference assignment before grading")
    data = read_json(root / "evaluation" / "dataset.json")
    if digest(data) != config["evaluation_data_digest"]:
        raise ValueError("Evaluation dataset changed after preparation")
    for arm, path in aggregate(root, config).items():
        command, run_id = evaluation_command(root, config, arm, path)
        if dry_run:
            print(json.dumps({"arm": arm, "command": command}, indent=2))
            continue
        check_evaluator()
        directory = root / "evaluation" / arm
        receipt_path = directory / "evaluation.json"
        env = docker_env()
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["run_id"] != run_id and not resume:
                raise ValueError(
                    "Predictions changed after evaluation started; do not reuse grades"
                )
            if not resume or (
                receipt["run_id"] == run_id
                and receipt["status"] == "finished"
                and evaluation_complete(directory, run_id, path)
            ):
                print(f"Skip evaluation {arm}: {receipt['status']}")
                continue
            print(f"Resume evaluation {arm}", flush=True)
            cleanup_evaluation_containers(config, receipt["run_id"], env)
            if receipt["run_id"] != run_id:
                archive_artifact(root, directory)
            else:
                # Keep official per-instance reports: the evaluator skips completed tasks.
                archive_artifact(root, receipt_path)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, {"run_id": run_id, "status": "running"})
        # Grading may download/build images, but never receives model service credentials.
        print(f"Evaluate {arm} (log directory: {directory})", flush=True)
        try:
            result = tracked_execute(
                root,
                command,
                cwd=directory,
                env=env,
                timeout=7200 * len(config["ids"]),
                limit=2_000_000,
            )
        finally:
            cleanup_evaluation_containers(config, run_id, env)
        (directory / "evaluator.log").write_text(result.output)
        atomic_json(
            receipt_path,
            {
                "run_id": run_id,
                "status": "finished" if result.exit_code == 0 else "evaluation_error",
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
        )


def evaluation_complete(directory: Path, run_id: str, predictions: Path) -> bool:
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        if not row["model_patch"].strip():
            continue
        model = row["model_name_or_path"].replace("/", "__")
        path = (
            directory / "logs/run_evaluation" / run_id / model / row["instance_id"] / "report.json"
        )
        if not path.exists():
            return False
        grade = read_json(path).get(row["instance_id"], {})
        if not isinstance(grade.get("resolved"), bool):
            return False
    return True


def report(root: Path, config: dict) -> dict:
    predictions = aggregate(root, config)  # verifies every sealed prediction before reading grades
    expected_runs = {
        arm: evaluation_command(root, config, arm, path)[1] for arm, path in predictions.items()
    }
    rows = []
    for iid in config["ids"]:
        for arm in config["arms"]:
            directory = attempt_path(root, iid, arm)
            state = (
                read_json(directory / "state.json")
                if (directory / "state.json").exists()
                else {"status": "not_attempted"}
            )
            result = (
                read_json(directory / "result.json") if (directory / "result.json").exists() else {}
            )
            prediction_path = directory / "prediction.jsonl"
            prediction = (
                json.loads(prediction_path.read_text()) if prediction_path.exists() else None
            )
            grade = None
            receipt = root / "evaluation" / arm / "evaluation.json"
            if receipt.exists() and prediction:
                run_id = read_json(receipt)["run_id"]
                if run_id != expected_runs[arm]:
                    raise ValueError("Evaluation receipt does not match frozen predictions")
                model = prediction["model_name_or_path"].replace("/", "__")
                path = (
                    root
                    / "evaluation"
                    / arm
                    / "logs"
                    / "run_evaluation"
                    / run_id
                    / model
                    / iid
                    / "report.json"
                )
                if path.exists():
                    grade = read_json(path).get(iid)
            outcome = (
                "resolved"
                if grade and grade.get("resolved") is True
                else (
                    "unresolved"
                    if grade and grade.get("resolved") is False
                    else (
                        "empty_patch"
                        if prediction and not prediction["model_patch"].strip()
                        else "not_graded"
                    )
                )
            )
            rows.append(
                {
                    "instance_id": iid,
                    "arm": arm,
                    "attempt_status": state["status"],
                    "agent_status": result.get("status"),
                    "outcome": outcome,
                    "resolved": outcome == "resolved",
                    "model_tokens": result.get("total_tokens"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "total_cost": None,
                }
            )
    preliminary = any(
        r["attempt_status"] in {"running", "not_attempted"}
        or (r["attempt_status"] == "generated" and r["outcome"] == "not_graded")
        for r in rows
    )
    summary = {
        "assigned_per_arm": len(config["ids"]),
        "arms": {},
        "rows": rows,
        "preliminary": preliminary,
        "note": "Unknown costs are not zero. Missing/ungraded attempts count as unsuccessful in assigned-task rates.",
    }
    for arm in config["arms"]:
        subset = [row for row in rows if row["arm"] == arm]
        solved = sum(row["resolved"] for row in subset)
        summary["arms"][arm] = {
            "resolved": solved,
            "assigned": len(subset),
            "resolved_rate": solved / len(subset),
            "graded": sum(row["outcome"] in ("resolved", "unresolved") for row in subset),
            "reported_model_tokens": sum(row["model_tokens"] or 0 for row in subset),
            "unknown_usage_attempts": sum(row["model_tokens"] is None for row in subset),
        }
    if set(config["arms"]) == {"baseline", "hydradb"}:
        outcomes = {(r["instance_id"], r["arm"]): r["resolved"] for r in rows}
        wins = sum(
            outcomes[iid, "hydradb"] and not outcomes[iid, "baseline"] for iid in config["ids"]
        )
        losses = sum(
            outcomes[iid, "baseline"] and not outcomes[iid, "hydradb"] for iid in config["ids"]
        )
        discordant = wins + losses
        p = (
            min(
                1.0,
                2
                * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
                / 2**discordant,
            )
            if discordant
            else 1.0
        )
        summary["paired"] = {
            "hydra_only": wins,
            "baseline_only": losses,
            "difference": (wins - losses) / len(config["ids"]),
            "mcnemar_exact_p": None if preliminary else p,
            "interpretation": "Operational subset result; not a confirmatory graph-effect claim.",
        }
    atomic_json(root / "report.json", summary)
    lines = [
        "# SWE-bench campaign report",
        "",
        "Status: "
        + (
            "PRELIMINARY — unfinished inference or grading."
            if preliminary
            else "Assignment outcomes accounted for."
        ),
        "",
        summary["note"],
        "",
        "| Arm | Resolved / assigned | Graded | Reported model tokens | Unknown-usage attempts |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm, stats in summary["arms"].items():
        lines.append(
            f"| {arm} | {stats['resolved']} / {stats['assigned']} | {stats['graded']} | {stats['reported_model_tokens']} | {stats['unknown_usage_attempts']} |"
        )
    lines.extend(
        [
            "",
            "Full per-task outcomes and failure categories: report.json.",
            "No success is inferred from an agent summary or evaluator process exit code.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary
