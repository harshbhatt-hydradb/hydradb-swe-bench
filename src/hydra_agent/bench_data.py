"""Pinned, allowlisted benchmark records and atomically replaced campaign metadata."""

import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

from .agent import Limits

DATASET = "princeton-nlp/SWE-bench_Verified"
EVALUATOR_VERSION = "3.0.15"
EVALUATOR_COMMIT = "b524f150d5d76f188c741d75669025f718c89c2e"
SAFE_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "version",
    "environment_setup_commit",
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def code_digest() -> str:
    root = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    for name in ("pyproject.toml", "uv.lock"):
        path = root.parent.parent / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(files)


def safe_task(row: dict) -> dict:
    task = {key: row[key] for key in SAFE_FIELDS if row.get(key) is not None}
    for key in SAFE_FIELDS[:-1]:
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError(f"Missing task field: {key}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", task["instance_id"]):
        raise ValueError("Unsafe instance ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", task["repo"]):
        raise ValueError("Unsafe repository name")
    if any(part in {".", ".."} for part in task["repo"].split("/")):
        raise ValueError("Unsafe repository name")
    for key in ("base_commit", "environment_setup_commit"):
        if key in task and (
            not isinstance(task[key], str) or not re.fullmatch(r"[a-f0-9]{40}", task[key])
        ):
            raise ValueError(f"Expected full commit SHA for {key}")
    return task


def prepare_campaign(
    root: Path,
    rows: list[dict],
    *,
    ids: list[str],
    dataset_revision: str,
    arms: list[str],
    seed: int,
    arch: str,
    deployment: str,
    endpoint: str,
    reasoning: str | None,
    limits: Limits,
    run_id: str | None = None,
) -> dict:
    if root.exists():
        raise ValueError("Campaign directory already exists; never overwrite a campaign")
    if not re.fullmatch(r"[a-f0-9]{40}", dataset_revision):
        raise ValueError("Dataset revision must resolve to a full commit SHA")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Choose a nonempty list of unique instance IDs")
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"baseline", "hydradb"}:
        raise ValueError("Arms must be baseline and/or hydradb without duplicates")
    if arch not in {"arm64", "x86_64"}:
        raise ValueError("Unsupported architecture")
    selected = [row for row in rows if row.get("instance_id") in ids]
    if len(selected) != len(ids) or len({r["instance_id"] for r in selected}) != len(ids):
        raise ValueError("Selected IDs must each occur exactly once in the pinned dataset")
    tasks = [safe_task(row) for row in selected]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    schedule = []
    for task in tasks:
        order = list(arms)
        rng.shuffle(order)
        schedule.extend({"instance_id": task["instance_id"], "arm": arm} for arm in order)
    config = {
        "schema": 1,
        "dataset": DATASET,
        "dataset_revision": dataset_revision,
        "split": "test",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_reference_commit": EVALUATOR_COMMIT,
        "ids": ids,
        "arms": arms,
        "seed": seed,
        "arch": arch,
        "schedule": schedule,
        "deployment": deployment,
        "endpoint": endpoint,
        "reasoning_effort": reasoning,
        "limits": asdict(limits),
        "retrieval_policy": "required_full_task_first_v1",
        "agent_digest": code_digest(),
        "tasks_digest": digest(tasks),
        "evaluation_data_digest": digest(selected),
        "index_limits": {"max_file_bytes": 500000, "max_total_bytes": 50000000, "timeout": 900},
    }
    if run_id is not None:
        config["run_id"] = run_id
    config["config_digest"] = digest(config)
    root.mkdir(parents=True)
    atomic_json(root / "inference" / "tasks.json", tasks)
    atomic_json(root / "evaluation" / "dataset.json", selected)
    atomic_json(root / "campaign.json", config)
    return config


def load_campaign(root: Path, *, check_code: bool = True) -> tuple[dict, list[dict]]:
    config = read_json(root / "campaign.json")
    if digest({k: v for k, v in config.items() if k != "config_digest"}) != config["config_digest"]:
        raise ValueError("Campaign configuration changed after preparation")
    tasks = read_json(root / "inference" / "tasks.json")
    if digest(tasks) != config["tasks_digest"] or any(safe_task(t) != t for t in tasks):
        raise ValueError("Inference task manifest changed or contains forbidden fields")
    if check_code and config["agent_digest"] != code_digest():
        raise ValueError("Agent or dependency configuration changed; prepare a new campaign")
    return config, tasks
