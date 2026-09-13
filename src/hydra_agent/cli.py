import argparse
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .agent import Limits, Trace, run_agent
from .config import AzureConfig, HydraConfig
from .environment import Workspace
from .hydradb import HydraMemory
from .indexing import build_corpus, reuse_corpus
from .interactive import chat_session
from .memory import MemoryScope
from .model import AzureModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Azure OpenAI coding-agent baseline")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    sub = parser.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser(
        "doctor", help="Validate configuration; --live checks Azure connectivity"
    )
    doctor.add_argument("--live", action="store_true")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--repo", type=Path, required=True)
    task = run.add_mutually_exclusive_group()
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    run.add_argument("--revision", default="HEAD")
    run.add_argument("--instance-id", default="local-task")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--backend", choices=("docker", "local"), default="docker")
    run.add_argument(
        "--allow-local-shell",
        action="store_true",
        help="Acknowledge that local model commands can access the host",
    )
    run.add_argument("--image", default="hydra-agent-sandbox:local")
    run.add_argument(
        "--memory",
        choices=("none", "hydradb"),
        default="none",
        help="hydradb uploads the selected commit and enables scoped graph retrieval",
    )
    run.add_argument("--index-timeout", type=int, default=900)
    run.add_argument(
        "--reuse-index",
        type=Path,
        help="Reuse a completed index-manifest.json; status checks and retrieval only, no uploads",
    )
    run.add_argument("--index-max-file-bytes", type=int, default=500_000)
    run.add_argument("--index-max-bytes", type=int, default=50_000_000)
    for field, default in asdict(Limits()).items():
        run.add_argument("--" + field.replace("_", "-"), type=int, default=default)
    sub.add_parser("run", parents=[run], help="Solve one issue in a disposable repository")
    sub.add_parser("chat", parents=[run], help="Interactive conversation in a persistent sandbox")
    args = parser.parse_args()
    if args.action == "run" and args.task is None and args.task_file is None:
        parser.error("run requires --task or --task-file")
    load_dotenv(args.env_file, override=False)
    model = None
    memory = None
    try:
        config = AzureConfig.from_env()
        if args.action == "doctor":
            print(f"Configuration valid. Deployment: {config.deployment}")
            if args.live:
                model = AzureModel(config)
                response = model.complete(
                    [{"role": "user", "content": "Reply OK."}], [], max_tokens=1024, timeout=30
                )
                print(
                    json.dumps(
                        {
                            "model": response.get("model"),
                            "usage": response.get("usage"),
                            "choices": response.get("choices"),
                        }
                    )
                )
            return 0
        if args.backend == "local" and not args.allow_local_shell:
            raise ValueError("Local execution requires --allow-local-shell; it is not a sandbox")
        limits = Limits(**{key: getattr(args, key) for key in asdict(Limits())})
        task_text = (
            args.task
            if args.task is not None
            else (args.task_file.read_text() if args.task_file else None)
        )
        if task_text is not None and not task_text.strip():
            raise ValueError("Task must not be empty")
        hydra_config = HydraConfig.from_env() if args.memory == "hydradb" else None
        if args.reuse_index and not hydra_config:
            raise ValueError("--reuse-index requires --memory hydradb")
        saved_index = json.loads(args.reuse_index.read_text()) if args.reuse_index else None
        if min(args.index_timeout, args.index_max_file_bytes, args.index_max_bytes) <= 0:
            raise ValueError("Indexing limits must be positive")
        attempt = uuid.uuid4().hex
        output = args.output.resolve() / attempt
        output.mkdir(parents=True, exist_ok=False)
        trace = Trace(
            output / "trajectory.jsonl",
            secrets=(config.api_key, hydra_config.api_key if hydra_config else ""),
        )
        print(f"Run artifacts: {output}", flush=True)
        metadata = {
            "attempt_id": attempt,
            "instance_id": args.instance_id,
            "started_at": datetime.now(UTC).isoformat(),
            "repository": str(args.repo.resolve()),
            "revision": args.revision,
            "deployment": config.deployment,
            "reasoning_effort": config.reasoning_effort,
            "backend": args.backend,
            "image": args.image if args.backend == "docker" else None,
            "memory": args.memory,
            "limits": asdict(limits),
            "sdk_max_retries": 2,
            "action": args.action,
        }
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            with Workspace(
                args.repo, revision=args.revision, backend=args.backend, image=args.image
            ) as workspace:
                metadata["base_commit"] = workspace.base_commit
                (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
                scope = MemoryScope(
                    str(args.repo.resolve()), workspace.base_commit, args.instance_id, attempt
                )
                if hydra_config:
                    if saved_index is not None:
                        corpus = reuse_corpus(
                            workspace.archive, scope, saved_index, hydra_config.database
                        )
                        memory = HydraMemory(
                            hydra_config,
                            corpus,
                            report=trace.emit,
                            existing_collection=saved_index["collection"],
                        )
                    else:
                        corpus = build_corpus(
                            workspace.archive,
                            scope,
                            max_file_bytes=args.index_max_file_bytes,
                            max_total_bytes=args.index_max_bytes,
                        )
                        memory = HydraMemory(hydra_config, corpus, report=trace.emit)
                    index_manifest = {
                        **corpus.manifest(),
                        **({"scope": saved_index["scope"]} if saved_index is not None else {}),
                        "database": hydra_config.database,
                        "collection": memory.collection,
                        "status": "pending",
                        "timeout_seconds": args.index_timeout,
                        "max_file_bytes": saved_index["max_file_bytes"]
                        if saved_index is not None
                        else args.index_max_file_bytes,
                        "max_total_bytes": saved_index["max_total_bytes"]
                        if saved_index is not None
                        else args.index_max_bytes,
                        "reused_from": str(args.reuse_index.resolve())
                        if args.reuse_index
                        else None,
                    }
                    index_path = output / "index-manifest.json"
                    index_path.write_text(json.dumps(index_manifest, indent=2) + "\n")
                    print(
                        (
                            "Checking existing graph (no uploads)"
                            if saved_index is not None
                            else f"Indexing {len(corpus.sources)} source files into HydraDB"
                        )
                        + f" ({hydra_config.database}/{memory.collection})...",
                        flush=True,
                    )
                    try:
                        index_manifest.update(memory.prepare(timeout=args.index_timeout))
                    except Exception:
                        index_manifest["status"] = "failed"
                        raise
                    finally:
                        index_path.write_text(json.dumps(index_manifest, indent=2) + "\n")
                    metadata["hydradb"] = {
                        "database": hydra_config.database,
                        "collection": memory.collection,
                        "graph_method": "automatic",
                        "mode": "thinking",
                        "max_search_calls": 40,
                        "retrieval_only": saved_index is not None,
                    }
                    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    print("Repository graph ready. Starting coding agent.", flush=True)
                model = AzureModel(config)
                if args.action == "chat":
                    return chat_session(
                        model,
                        workspace,
                        limits,
                        trace,
                        output,
                        memory=memory,
                        scope=scope if memory else None,
                        initial_task=task_text,
                    )
                result = run_agent(
                    model,
                    workspace,
                    task_text,
                    limits,
                    trace,
                    memory=memory,
                    scope=scope if memory else None,
                )
                patch = workspace.patch()
                (output / "patch.diff").write_text(patch)
                prediction = {
                    "instance_id": args.instance_id,
                    "model_name_or_path": ("hydra-baseline/" if memory is None else "hydra-graph/")
                    + config.deployment,
                    "model_patch": patch,
                }
                (output / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
                result["patch_bytes"] = len(patch.encode())
                (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "submitted" else 2
        except Exception as exc:
            trace.emit("run_error", error_type=type(exc).__name__)
            (output / "result.json").write_text(
                json.dumps({"status": "run_error", "error_type": type(exc).__name__}) + "\n"
            )
            raise
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- CLI failure boundary
        # API / subprocess exception messages can contain credentials or source text.
        print(
            f"Run failed ({type(exc).__name__}). Check Azure configuration, Git revision, "
            "and Docker availability; see run artifacts if created.",
            file=sys.stderr,
        )
        if isinstance(exc, subprocess.CalledProcessError):
            print(f"Command exit code: {exc.returncode}", file=sys.stderr)
        return 2
    finally:
        if memory is not None:
            memory.close()
        if model is not None:
            model.close()


if __name__ == "__main__":
    sys.exit(main())
