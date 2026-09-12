"""Build a reproducible upload corpus from the exact archive used by the execution workspace."""

import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from .memory import MemoryScope

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sql",
    ".sh",
    ".bash",
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".graphql",
    ".proto",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".next",
    ".ssh",
}
SECRET_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
}


@dataclass(frozen=True)
class Source:
    id: str
    path: str
    sha256: str
    size: int
    text: str = field(repr=False)

    def manifest(self) -> dict:
        return {"id": self.id, "path": self.path, "sha256": self.sha256, "bytes": self.size}


@dataclass
class Corpus:
    scope: MemoryScope
    sources: list[Source]
    excluded: list[dict]

    def manifest(self) -> dict:
        return {
            "version": 1,
            "scope": asdict(self.scope),
            "graph_method": "hydradb_automatic",
            "sources": [source.manifest() for source in self.sources],
            "excluded": self.excluded,
            "source_count": len(self.sources),
            "total_bytes": sum(source.size for source in self.sources),
        }


def build_corpus(
    archive: Path,
    scope: MemoryScope,
    *,
    max_file_bytes: int = 500_000,
    max_total_bytes: int = 50_000_000,
) -> Corpus:
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Index byte limits must be positive")
    sources, excluded = [], []
    total = 0
    with tarfile.open(archive) as tree:
        for member in tree:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            reason = None
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                reason = "non_regular_or_unsafe_path"
            elif any(part in SKIP_PARTS for part in path.parts):
                reason = "excluded_directory"
            elif (
                path.name.lower() in SECRET_NAMES
                or path.name.lower().startswith(".env.")
                or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
            ):
                reason = "credential_filename"
            elif path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {
                "Dockerfile",
                "Makefile",
                "LICENSE",
                "Gemfile",
            }:
                reason = "unsupported_extension"
            elif member.size > max_file_bytes:
                reason = "oversized_file"
            if reason:
                excluded.append({"path": member.name, "reason": reason})
                continue
            stream = tree.extractfile(member)
            assert stream is not None
            raw = stream.read(max_file_bytes + 1)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                excluded.append({"path": member.name, "reason": "non_utf8"})
                continue
            if b"\0" in raw or not content.strip():
                excluded.append({"path": member.name, "reason": "binary_or_empty"})
                continue
            total += len(raw)
            if total > max_total_bytes:
                raise ValueError("Repository exceeds --index-max-bytes; no data has been uploaded")
            digest = hashlib.sha256(raw).hexdigest()
            identity = json.dumps([asdict(scope), member.name, digest], sort_keys=True).encode()
            source_id = "src_" + hashlib.sha256(identity).hexdigest()
            sources.append(Source(source_id, member.name, digest, len(raw), content))
    if not sources:
        raise ValueError("No eligible UTF-8 source files in the selected snapshot")
    return Corpus(scope, sources, excluded)


def reuse_corpus(archive: Path, scope: MemoryScope, manifest: dict, database: str) -> Corpus:
    """Verify a saved index against the local snapshot, retaining its remote source IDs."""
    if manifest.get("version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Reuse requires a completed version-1 index manifest")
    if manifest.get("database") != database:
        raise ValueError("Reuse manifest database differs from HYDRA_DB_DATABASE")
    old_scope = MemoryScope(**manifest["scope"])
    if (old_scope.repository, old_scope.base_commit) != (scope.repository, scope.base_commit):
        raise ValueError("Reuse manifest repository/commit does not match --repo and --revision")
    if manifest.get("collection") != "attempt_" + old_scope.attempt_id:
        raise ValueError("Reuse manifest collection does not match its original attempt")
    original = build_corpus(
        archive,
        old_scope,
        max_file_bytes=manifest["max_file_bytes"],
        max_total_bytes=manifest["max_total_bytes"],
    )
    if original.manifest()["sources"] != manifest.get("sources"):
        raise ValueError("Reuse manifest source IDs/hashes do not match the repository snapshot")
    return Corpus(scope, original.sources, original.excluded)
