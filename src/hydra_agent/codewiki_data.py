"""Pinned CodeWikiBench data and code-only documentation inputs."""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

import httpx

from .bench_data import atomic_json, digest, read_json
from .indexing import Corpus, build_corpus
from .memory import MemoryScope

DATASET = "anhnh2002/codewikibench"
DATASET_REVISION = "6d215eb7d50a164e370a9a5703b813f9da345965"
EVALUATOR_REPO = "https://github.com/FSoft-AI4Code/CodeWikiBench.git"
EVALUATOR_REVISION = "5e728fb40492effb54d59041f908dbf9079fe238"
REPOSITORIES = (
    "Chart.js",
    "FluentValidation",
    "OpenHands",
    "electron",
    "git-credential-manager",
    "graphrag",
    "json",
    "libsql",
    "logstash",
    "marktext",
    "material-components-android",
    "mermaid",
    "ml-agents",
    "puppeteer",
    "qmk_firmware",
    "rasa",
    "storybook",
    "sumatrapdf",
    "svelte",
    "trino",
    "wazuh",
    "x64dbg",
)
POLICY = "source_build_tests_no_docs_or_fixtures_v1"
DOC_PARTS = {"docs", "doc", "documentation", "wiki", "website", "site", ".github"}
PROSE = {".md", ".mdx", ".rst", ".txt"}


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE, timeout=300
    ).strip()


def ensure_checkout(path: Path, url: str, revision: str) -> None:
    """Fetch one exact commit; never execute repository code or touch an existing checkout."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full commit SHA is required")
    if not path.exists():
        path.mkdir(parents=True)
        git("init", str(path))
        git("remote", "add", "origin", url, cwd=path)
    if git("remote", "get-url", "origin", cwd=path).removesuffix(".git") != url.removesuffix(
        ".git"
    ):
        raise ValueError("Cached repository origin does not match")
    try:
        resolved = git("rev-parse", "--verify", revision + "^{commit}", cwd=path)
    except subprocess.CalledProcessError:
        git("fetch", "--depth", "1", "origin", revision, cwd=path)
        resolved = git("rev-parse", "--verify", revision + "^{commit}", cwd=path)
    if resolved != revision:
        raise ValueError("Repository commit verification failed")


def safe_metadata(record: dict) -> dict:
    raw = record.get("metadata", record)
    result = {k: raw[k] for k in ("repo_name", "repo_url", "commit_id")}
    if result["repo_name"] not in REPOSITORIES:
        raise ValueError("Unknown CodeWikiBench repository")
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", result["repo_url"]):
        raise ValueError("Unexpected repository URL")
    if not re.fullmatch(r"[a-f0-9]{40}", result["commit_id"]):
        raise ValueError("Invalid benchmark commit")
    return result


def prepare_record(root: Path, name: str, cache: Path) -> dict:
    if name not in REPOSITORIES:
        raise ValueError("Unknown benchmark repository")
    evaluation = root / "evaluation" / "reference.json"
    if not evaluation.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/raw_data/{name}.json"
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        record = response.json()
        metadata = safe_metadata(record)
        if metadata["repo_name"] != name:
            raise ValueError("Downloaded record is for the wrong repository")
        atomic_json(evaluation, record)
    record = read_json(evaluation)
    metadata = safe_metadata(record)
    provenance_path = root / "provenance.json"
    provenance = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "record_digest": digest(record),
        "metadata": metadata,
    }
    if provenance_path.exists():
        saved = read_json(provenance_path)
        if any(saved.get(key) != value for key, value in provenance.items()):
            raise ValueError("Saved benchmark record differs from its recorded provenance")
    rubrics = record["rubrics"]
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)
    atomic_json(root / "evaluation" / "rubrics.json", rubrics)
    # Only these three fields cross into the generation task. No reference docs/tree/rubrics.
    atomic_json(root / "inference" / "task.json", metadata)
    archive = root / "inference" / "snapshot.tar"
    if not archive.exists():
        checkout = cache / name
        ensure_checkout(checkout, metadata["repo_url"], metadata["commit_id"])
        temp = archive.with_suffix(".tmp")
        git(
            "archive",
            "--format=tar",
            f"--output={temp.resolve()}",
            metadata["commit_id"],
            cwd=checkout,
        )
        temp.replace(archive)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if provenance_path.exists() and saved.get("snapshot_sha256") != archive_digest:
        raise ValueError("Pinned repository archive changed")
    provenance["snapshot_sha256"] = archive_digest
    atomic_json(provenance_path, provenance)
    return metadata


def code_corpus(archive: Path, scope: MemoryScope, max_bytes: int = 50_000_000) -> Corpus:
    corpus = build_corpus(archive, scope, max_total_bytes=max_bytes)
    allowed, excluded = [], list(corpus.excluded)
    for source in corpus.sources:
        path = PurePosixPath(source.path)
        if path.suffix.lower() in PROSE or any(p.lower() in DOC_PARTS for p in path.parts):
            excluded.append({"path": source.path, "reason": "reference_documentation_policy"})
        elif any(p.lower() in {"fixtures", "__fixtures__", "__snapshots__"} for p in path.parts):
            excluded.append({"path": source.path, "reason": "test_fixture_data"})
        elif path.name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock"}:
            excluded.append({"path": source.path, "reason": "dependency_lockfile"})
        else:
            allowed.append(source)
    if not allowed:
        raise ValueError("No eligible code/build/test files")
    return Corpus(scope, allowed, excluded)


def load_official_evaluator(cache: Path) -> dict:
    """Load only the published prompt and pure scoring functions, without upstream SDKs."""
    ensure_checkout(cache, EVALUATOR_REPO, EVALUATOR_REVISION)
    source = git("show", f"{EVALUATOR_REVISION}:src/judge/judge.py", cwd=cache)
    tree = ast.parse(source)
    names = {"is_leaf_node", "collect_leaf_requirements", "calculate_scores_bottom_up"}
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "EVALUATION_SYSTEM_PROMPT" for t in node.targets
            )
        )
    ]
    namespace = {"json": json}
    exec(  # noqa: S102 -- selected nodes from the pinned upstream commit
        compile(ast.Module(body=selected, type_ignores=[]), "pinned_codewikibench_judge", "exec"),
        namespace,
    )
    if not names.issubset(namespace) or "EVALUATION_SYSTEM_PROMPT" not in namespace:
        raise ValueError("Pinned evaluator interface changed")
    namespace["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return namespace


def validate_rubrics(rubrics: list) -> None:
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError("Rubrics must be nonempty")
    for node in rubrics:
        if not isinstance(node.get("requirements"), str) or not node["requirements"].strip():
            raise ValueError("Missing rubric requirement")
        weight = node.get("weight")
        if type(weight) not in (int, float) or not 0 < weight < float("inf"):
            raise ValueError("Invalid rubric weight")
        if node.get("sub_tasks"):
            validate_rubrics(node["sub_tasks"])


def generation_fingerprint() -> str:
    return digest(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("codewiki*.py"))
        }
    )
