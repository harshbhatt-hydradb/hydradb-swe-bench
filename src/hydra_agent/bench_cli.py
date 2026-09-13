"""Run SWE-bench with automatic setup and resumable results."""

import argparse
import fcntl
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .agent import Limits
from .bench_data import (
    DATASET,
    atomic_json,
    code_digest,
    load_campaign,
    prepare_campaign,
    read_json,
)
from .bench_environment import build_task, check_evaluator, preflight
from .bench_process import recover_child, run_lock
from .bench_runner import evaluate, recover_containers, report, run_campaign
from .config import AzureConfig, HydraConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="run", choices=["run", "evaluate", "report", "stop", "doctor"]
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output", type=Path, default=Path(os.environ.get("HYDRA_BENCH_OUTPUT", "runs/benchmark"))
    )
    parser.add_argument("--restart", action="store_true", help="Stop the active run and resume")
    parser.add_argument("--fresh", action="store_true", help="Archive results and start a new run")
    parser.add_argument("--instances", nargs="+", help="Task IDs; initially sympy__sympy-20590")
    parser.add_argument("--arms", nargs="+", choices=["baseline", "hydradb"])
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--arch", choices=["x86_64", "arm64"], help="Defaults to native architecture"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--min-free-gb", type=float, default=120)
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview grading commands with evaluate"
    )
    args = parser.parse_args(argv)
    if args.fresh and args.action != "run":
        parser.error("--fresh is only available when running the benchmark")
    if args.dry_run and args.action != "evaluate":
        parser.error("--dry-run is only available with evaluate")
    if args.min_free_gb <= 0:
        parser.error("--min-free-gb must be positive")
    for name in ("max_steps", "max_total_tokens", "wall_seconds"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    return args


def settings(args, saved: dict) -> dict:
    limits = saved.get("limits", {})
    native = {"aarch64": "arm64", "amd64": "x86_64"}.get(platform.machine(), platform.machine())
    defaults = {
        "ids": ["sympy__sympy-20590"],
        "arms": ["baseline", "hydradb"],
        "dataset_revision": "main",
        "arch": native,
        "seed": 42,
    }
    result = {}
    for key, default in defaults.items():
        value = getattr(args, "instances" if key == "ids" else key)
        result[key] = value if value is not None else saved.get(key, default)
    result["limits"] = {
        **vars(Limits()),
        **limits,
        **{
            key: getattr(args, key)
            for key in ("max_steps", "max_total_tokens", "wall_seconds")
            if getattr(args, key) is not None
        },
    }
    azure = AzureConfig.from_env()
    result.update(
        endpoint=azure.endpoint,
        deployment=azure.deployment,
        reasoning_effort=azure.reasoning_effort,
    )
    if "hydradb" in result["arms"]:
        HydraConfig.from_env()
    return result


def download_tasks(revision):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(DATASET, revision=revision).sha
    return resolved, list(load_dataset(DATASET, revision=resolved, split="test"))


def ensure_run(root: Path, args) -> tuple[dict, list[dict]]:
    saved, _ = (
        load_campaign(root, check_code=False) if (root / "campaign.json").exists() else ({}, [])
    )
    if not saved and root.exists() and any(root.iterdir()):
        raise ValueError("Output directory contains unrelated files; choose an empty --output")
    desired = settings(args, saved)
    changed = saved and (
        saved["agent_digest"] != code_digest()
        or any(saved[key] != value for key, value in desired.items())
    )
    if saved and not args.fresh and not changed:
        return load_campaign(root)
    check_evaluator()
    if (
        saved
        and desired["dataset_revision"] == saved["dataset_revision"]
        and set(desired["ids"]) <= set(saved["ids"])
    ):
        rows = read_json(root / "evaluation/dataset.json")
        from .bench_data import digest

        if digest(rows) != saved["evaluation_data_digest"]:
            raise ValueError("Saved evaluation dataset changed")
        revision = saved["dataset_revision"]
    else:
        print("Loading benchmark tasks…", flush=True)
        revision, rows = download_tasks(desired["dataset_revision"])
    staging = root.with_name("." + root.name + "-prepare-" + uuid.uuid4().hex)
    config = prepare_campaign(
        staging,
        rows,
        ids=desired["ids"],
        arms=desired["arms"],
        dataset_revision=revision,
        seed=desired["seed"],
        arch=desired["arch"],
        endpoint=desired["endpoint"],
        deployment=desired["deployment"],
        reasoning=desired["reasoning_effort"],
        limits=Limits(**desired["limits"]),
        run_id=uuid.uuid4().hex,
    )
    if root.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        archive = root.with_name(root.name + "-" + stamp + "-" + uuid.uuid4().hex[:6])
        root.rename(archive)
        print(f"Previous results saved: {archive}", flush=True)
    staging.rename(root)
    print(f"Prepared {len(config['ids'])} task(s), arms: {', '.join(config['arms'])}", flush=True)
    return load_campaign(root)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file, override=False)
    root = args.output.resolve()
    try:
        with run_lock(root, restart=args.restart, stop=args.action == "stop", recover=False):
            # Coordinate with the previous CLI while old commands are still supported.
            legacy_lock = None
            try:
                if (root / ".pipeline.lock").exists():
                    legacy_lock = (root / ".pipeline.lock").open("r")
                    try:
                        fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise ValueError(
                            "An older pipeline is using this output; stop it with Ctrl-C first"
                        ) from None
                recover_child(root)
                if (
                    args.action in {"run", "evaluate", "stop"}
                    and not args.dry_run
                    and (root / "campaign.json").exists()
                ):
                    recover_containers(root, load_campaign(root, check_code=False)[0])
                if args.action == "stop":
                    print("Benchmark stopped. Run ./benchmark to resume.")
                    return 0
                if args.action == "run":
                    config, tasks = ensure_run(root, args)
                    if legacy_lock is None or not (root / ".pipeline.lock").exists():
                        if legacy_lock is not None:
                            legacy_lock.close()
                        legacy_lock = (root / ".pipeline.lock").open("a+")
                        try:
                            fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            raise ValueError(
                                "An older pipeline started on this output; stop it first"
                            ) from None
                else:
                    if not (root / "campaign.json").exists():
                        raise ValueError("No saved benchmark yet. Run ./benchmark first.")
                    config, tasks = load_campaign(root, check_code=False)
                print(f"Results: {root}", flush=True)
                if args.action in {"run", "doctor", "evaluate"} and not args.dry_run:
                    check_evaluator()
                    status = preflight(root, config["arch"], min_free_gb=args.min_free_gb)
                    atomic_json(root / "preflight.json", status)
                if args.action == "run":
                    for task in tasks:
                        print(f"Prepare environment: {task['instance_id']}", flush=True)
                        build_task(root, task, config["arch"])
                    run_campaign(root, config, tasks, args.env_file.resolve(), resume=True)
                if args.action in {"run", "evaluate"}:
                    evaluate(root, config, dry_run=args.dry_run, resume=True)
                if args.action in {"run", "evaluate", "report"} and not args.dry_run:
                    result = report(root, config)
                    print(json.dumps(result["arms"], indent=2))
                    print(f"Report: {root / 'report.md'}")
                    if args.action != "report" and (
                        result["preliminary"]
                        or any(row["attempt_status"] != "generated" for row in result["rows"])
                    ):
                        print("Some work is unfinished. Run ./benchmark again to resume.")
                        return 2
                return 0
            finally:
                if legacy_lock is not None:
                    legacy_lock.close()
    except KeyboardInterrupt:
        print("Stopped. Progress saved; run ./benchmark to resume.")
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"Benchmark: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- avoid exposing service errors or credentials
        print(f"Benchmark failed ({type(exc).__name__}). Logs: {root}. Run ./benchmark to resume.")
        return 2
