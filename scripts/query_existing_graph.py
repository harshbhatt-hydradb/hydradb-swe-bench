"""Read an existing repository graph without ingestion, a coding model, or a container."""

import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

from hydra_agent.config import HydraConfig
from hydra_agent.environment import git
from hydra_agent.hydradb import HydraMemory
from hydra_agent.indexing import reuse_corpus
from hydra_agent.memory import MemoryScope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reuse-index", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    load_dotenv(".env", override=False)
    config = HydraConfig.from_env()
    saved = json.loads(args.reuse_index.read_text())
    repo = args.repo.resolve()
    commit = git(repo, "rev-parse", "--verify", "--end-of-options", f"{args.revision}^{{commit}}")
    scope = MemoryScope(str(repo), commit, "retrieval-only", uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix="hydra-reuse-check-") as directory:
        archive = Path(directory) / "repo.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", "--output", str(archive), commit],
            check=True,
            timeout=60,
        )
        corpus = reuse_corpus(archive, scope, saved, config.database)
    events = []
    memory = HydraMemory(
        config,
        corpus,
        existing_collection=saved["collection"],
        report=lambda event, **data: events.append({"event": event, **data}),
    )
    try:
        status = memory.prepare(timeout=120)
        hits = memory.search(args.query, scope=scope, limit=8)
        print(json.dumps({"index": status, "hits": hits, "events": events}, indent=2))
    finally:
        memory.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- never print raw HTTP exceptions or credentials
        raise SystemExit(f"Existing graph query failed ({type(exc).__name__})") from None
