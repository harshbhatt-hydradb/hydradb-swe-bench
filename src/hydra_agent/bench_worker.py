"""One isolated benchmark inference attempt. Never loads evaluation dataset rows."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import Limits, Trace, run_agent
from .bench_data import atomic_json, safe_task
from .config import AzureConfig, HydraConfig
from .environment import Workspace
from .hydradb import HydraMemory
from .indexing import build_corpus
from .memory import MemoryScope
from .model import AzureModel


def run_job(job: dict) -> None:
    if job["arm"] not in {"baseline", "hydradb"}:
        raise ValueError("Unknown benchmark arm")
    task = safe_task(job["task"])
    if task != job["task"]:
        raise ValueError("Worker job contains forbidden task fields")
    output = Path(job["output"])
    azure = AzureConfig.from_env()
    if (azure.endpoint, azure.deployment, azure.reasoning_effort) != (
        job["endpoint"],
        job["deployment"],
        job["reasoning_effort"],
    ):
        raise ValueError("Azure configuration differs from the frozen campaign")
    hydra = HydraConfig.from_env() if job["arm"] == "hydradb" else None
    trace = Trace(output / "trajectory.jsonl", (azure.api_key, hydra.api_key if hydra else ""))
    memory = model = None
    try:
        with Workspace(
            Path(job["repository"]),
            revision=task["base_commit"],
            image=job["image"],
            seed_from_image=True,
            container_name=job["container"],
        ) as workspace:
            scope = MemoryScope(
                task["repo"], task["base_commit"], task["instance_id"], job["attempt_id"]
            )
            if hydra:
                corpus = build_corpus(
                    workspace.archive,
                    scope,
                    max_file_bytes=job["index_limits"]["max_file_bytes"],
                    max_total_bytes=job["index_limits"]["max_total_bytes"],
                )
                memory = HydraMemory(hydra, corpus, report=trace.emit)
                manifest = {
                    **corpus.manifest(),
                    "database": hydra.database,
                    "collection": memory.collection,
                    "status": "pending",
                    **job["index_limits"],
                }
                atomic_json(output / "index-manifest.json", manifest)
                try:
                    manifest.update(memory.prepare(timeout=job["index_limits"]["timeout"]))
                except Exception:
                    manifest["status"] = "failed"
                    raise
                finally:
                    atomic_json(output / "index-manifest.json", manifest)
            model = AzureModel(azure)
            result = run_agent(
                model,
                workspace,
                task["problem_statement"],
                Limits(**job["limits"]),
                trace,
                memory=memory,
                scope=scope if memory else None,
            )
            patch = workspace.patch()
            (output / "patch.diff").write_text(patch)
            prediction = {
                "instance_id": task["instance_id"],
                "model_name_or_path": "hydra-" + job["arm"] + "/" + azure.deployment,
                "model_patch": patch,
            }
            (output / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
            atomic_json(output / "result.json", {**result, "patch_bytes": len(patch.encode())})
    finally:
        if memory:
            memory.close()
        if model:
            model.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    job = json.loads(args.job.read_text())
    try:
        run_job(job)
    except Exception as exc:  # noqa: BLE001 -- sanitized boundary, no keys or API response text
        atomic_json(
            Path(job["output"]) / "result.json",
            {"status": "run_error", "error_type": type(exc).__name__},
        )
        print(f"Attempt failed: {type(exc).__name__}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
