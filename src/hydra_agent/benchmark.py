"""SWE-bench pipeline CLI. Defaults to serial execution and explicit task selection."""

import argparse
import fcntl
import json
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import Limits
from .bench_data import DATASET, atomic_json, load_campaign, prepare_campaign
from .bench_environment import build_task, check_evaluator, preflight
from .bench_runner import evaluate, report, run_campaign
from .config import AzureConfig, HydraConfig


def main() -> int:
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        if "--campaign" not in sys.argv[1:] and not any(
            arg.startswith("--campaign=") for arg in sys.argv[1:]
        ):
            from .bench_cli import main as simple_main

            return simple_main()
        return legacy_main()
    finally:
        signal.signal(signal.SIGTERM, previous)


def legacy_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser(
        "prepare",
        help="Download and freeze explicit tasks, inference records, and evaluator-only data",
    )
    prepare.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--instances", nargs="+", required=True)
    prepare.add_argument(
        "--dataset-revision",
        default="main",
        help="Resolved to an immutable HF dataset commit before loading",
    )
    prepare.add_argument(
        "--arms", nargs="+", choices=["baseline", "hydradb"], default=["baseline", "hydradb"]
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--arch", choices=["x86_64", "arm64"], default="x86_64")
    prepare.add_argument("--max-steps", type=int, default=40)
    prepare.add_argument("--max-total-tokens", type=int, default=200000)
    prepare.add_argument("--wall-seconds", type=int, default=1200)
    for name in ("doctor", "build", "run", "evaluate", "report", "pipeline"):
        command = sub.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name in {"doctor", "build", "evaluate", "pipeline"}:
            command.add_argument("--min-free-gb", type=float, default=120)
        if name == "evaluate":
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    root = args.campaign.resolve()
    try:
        if args.action == "prepare":
            if root.exists():
                raise ValueError("Campaign already exists; use a new directory")
            check_evaluator()
            from datasets import load_dataset
            from huggingface_hub import HfApi

            azure = AzureConfig.from_env()
            if "hydradb" in args.arms:
                HydraConfig.from_env()
            revision = HfApi().dataset_info(DATASET, revision=args.dataset_revision).sha
            rows = list(load_dataset(DATASET, revision=revision, split="test"))
            config = prepare_campaign(
                root,
                rows,
                ids=args.instances,
                dataset_revision=revision,
                arms=args.arms,
                seed=args.seed,
                arch=args.arch,
                deployment=azure.deployment,
                endpoint=azure.endpoint,
                reasoning=azure.reasoning_effort,
                limits=Limits(
                    max_steps=args.max_steps,
                    max_total_tokens=args.max_total_tokens,
                    wall_seconds=args.wall_seconds,
                ),
            )
            print(
                json.dumps(
                    {
                        "campaign": str(root),
                        "dataset_revision": revision,
                        "instances": config["ids"],
                        "arms": config["arms"],
                    },
                    indent=2,
                )
            )
            return 0
        config, tasks = load_campaign(root, check_code=args.action not in {"report", "doctor"})
        with (root / ".pipeline.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another process is operating on this campaign") from None
            dry_run = getattr(args, "dry_run", False)
            if args.action in {"doctor", "build", "pipeline", "evaluate"} and not dry_run:
                if args.min_free_gb <= 0:
                    raise ValueError("--min-free-gb must be positive")
                check_evaluator()
                status = preflight(root, config["arch"], min_free_gb=args.min_free_gb)
                atomic_json(root / "preflight.json", status)
                print(json.dumps(status, indent=2))
            if args.action in {"build", "pipeline"}:
                for task in tasks:
                    print(f"Prepare environment: {task['instance_id']}", flush=True)
                    build_task(root, task, config["arch"])
            if args.action in {"run", "pipeline"}:
                azure = AzureConfig.from_env()
                if (azure.endpoint, azure.deployment, azure.reasoning_effort) != (
                    config["endpoint"],
                    config["deployment"],
                    config["reasoning_effort"],
                ):
                    raise ValueError("Model configuration differs from the campaign")
                if "hydradb" in config["arms"]:
                    HydraConfig.from_env()
                missing = [
                    task["instance_id"]
                    for task in tasks
                    if not (root / "build" / task["instance_id"] / "environment.json").exists()
                ]
                if missing:
                    raise ValueError("Build environments first; missing: " + ", ".join(missing))
                run_campaign(root, config, tasks, args.env_file.resolve())
            if args.action in {"evaluate", "pipeline"}:
                evaluate(root, config, dry_run=dry_run)
            if args.action in {"report", "pipeline", "run", "evaluate"} and not dry_run:
                result = report(root, config)
                print(json.dumps(result["arms"], indent=2))
                print(f"Report: {root / 'report.md'}")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Pipeline blocked: {exc}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Attempts are retained; inference is never automatically repeated.")
        return 130
    except Exception as exc:  # noqa: BLE001 -- avoid printing raw API errors/credentials
        print(f"Pipeline failed ({type(exc).__name__}); inspect campaign logs.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
