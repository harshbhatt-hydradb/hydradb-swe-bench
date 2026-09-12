"""Memory provider contract; hosted HydraDB implementation lives in hydradb.py."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryScope:
    repository: str
    base_commit: str
    instance_id: str
    attempt_id: str


class MemoryProvider(Protocol):
    def search(self, query: str, *, scope: MemoryScope, limit: int) -> list[dict]: ...


class NullMemory:
    def search(self, query: str, *, scope: MemoryScope, limit: int) -> list[dict]:
        return []
