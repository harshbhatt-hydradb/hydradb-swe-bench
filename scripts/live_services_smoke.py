"""Exercise real HydraDB ingestion/retrieval and Azure tools without executing generated code."""

import argparse
import io
import json
import tarfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

from hydra_agent.agent import Limits, Trace, run_agent
from hydra_agent.config import AzureConfig, HydraConfig
from hydra_agent.environment import CommandResult
from hydra_agent.hydradb import HydraMemory
from hydra_agent.indexing import build_corpus
from hydra_agent.memory import MemoryScope
from hydra_agent.model import AzureModel


class NoExecutionWorkspace:
    """The service smoke test deliberately has no command execution capability."""

    def changed_paths(self) -> set[str]:
        return set()

    def run(self, command: str, timeout: float) -> CommandResult:
        return CommandResult("Command execution is disabled in this service smoke test.", 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-timeout", type=int, default=300)
    args = parser.parse_args()
    load_dotenv(".env", override=False)
    azure_config = AzureConfig.from_env()
    hydra_config = HydraConfig.from_env()
    attempt = uuid.uuid4().hex
    output = Path("runs") / ("services-smoke-" + attempt)
    output.mkdir(parents=True, exist_ok=False)
    trace = Trace(output / "trajectory.jsonl", secrets=(azure_config.api_key, hydra_config.api_key))
    scope = MemoryScope("synthetic-services-smoke", "fixture-v1", "services-smoke", attempt)
    archive = output / "fixture.tar"
    with tarfile.open(archive, "w") as tree:
        content = (
            b"# Lantern calculator\n\n"
            b"The Lantern calculator uses the add_numbers function in arithmetic.py.\n"
            b"Its repository verification phrase is AMBER-PINE-427.\n"
            b"The addition contract is add_numbers(2, 3) == 5.\n"
        )
        member = tarfile.TarInfo("README.md")
        member.size = len(content)
        tree.addfile(member, io.BytesIO(content))
    corpus = build_corpus(archive, scope)
    memory = HydraMemory(hydra_config, corpus, report=trace.emit)
    model = None
    manifest = {
        **corpus.manifest(),
        "database": hydra_config.database,
        "collection": memory.collection,
        "deployment": azure_config.deployment,
        "test_type": "live_services_no_command_execution",
        "status": "pending",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Live service smoke artifacts: {output.resolve()}", flush=True)
    result = {"passed": False, "status": "interrupted"}
    try:
        print("Waiting for HydraDB provisioning and source graph completion...", flush=True)
        manifest["indexing"] = memory.prepare(timeout=args.index_timeout)
        manifest["status"] = "indexed"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        hits = memory.search(
            "Lantern calculator repository verification phrase", scope=scope, limit=3
        )
        trace.emit("direct_retrieval", hits=hits)
        if not any("AMBER-PINE-427" in hit["text"] for hit in hits):
            raise RuntimeError("Live retrieval did not return the uploaded fixture evidence")
        print("HydraDB ingestion and retrieval passed. Testing the Azure tool loop...", flush=True)
        model = AzureModel(azure_config)
        result = run_agent(
            model,
            NoExecutionWorkspace(),
            "This is a service integration smoke test, not a repair task. No shell execution is "
            "available. Use memory_search to find the Lantern calculator repository verification "
            "phrase, then call finish with that phrase and the addition contract in your summary. "
            "Do not guess the phrase: retrieve it from repository evidence.",
            Limits(
                max_steps=6, max_total_tokens=40000, max_completion_tokens=2048, wall_seconds=180
            ),
            trace,
            memory=memory,
            scope=scope,
        )
        events = [
            json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()
        ]
        called_search = any(
            event.get("event") == "model"
            and any(
                call.get("function", {}).get("name") == "memory_search"
                for choice in event["response"].get("choices", [])
                for call in choice.get("message", {}).get("tool_calls", [])
            )
            for event in events
        )
        result["passed"] = (
            result["status"] == "submitted"
            and called_search
            and "AMBER-PINE-427" in result["summary"]
        )
        result["scope"] = {"database": hydra_config.database, "collection": memory.collection}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    except Exception as exc:  # noqa: BLE001 -- report failures without leaking credentials
        result = {"passed": False, "status": "service_error", "error_type": type(exc).__name__}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        trace.emit("service_error", error_type=type(exc).__name__)
        print(json.dumps(result), flush=True)
        return 2
    finally:
        manifest["status"] = "passed" if result.get("passed") else "failed"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        memory.close()
        if model is not None:
            model.close()


if __name__ == "__main__":
    raise SystemExit(main())
