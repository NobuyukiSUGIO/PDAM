"""Memory Store facade (§7.1) wrapping an adapter and lineage bookkeeping."""
from __future__ import annotations

from typing import Optional

from ..schema import State
from .base import MemoryAdapter
from .kv import KVMemoryAdapter
from .summary import SummaryMemoryAdapter
from .vector import VectorMemoryAdapter

ADAPTERS = {
    "vector": VectorMemoryAdapter,
    "summary": SummaryMemoryAdapter,
    "kv": KVMemoryAdapter,
}


def make_adapter(kind: str) -> MemoryAdapter:
    if kind not in ADAPTERS:
        raise ValueError(f"unknown memory kind {kind!r}; choose {list(ADAPTERS)}")
    return ADAPTERS[kind]()


class MemoryStore:
    """Responsibility: storing conversations, summaries, success cases and profiles (§7.1)."""

    def __init__(self, kind: str = "vector") -> None:
        self.adapter = make_adapter(kind)
        self.kind = kind

    def write(self, state: State) -> State:
        self.adapter.add(state)
        return state

    def get(self, state_id: str) -> Optional[State]:
        return self.adapter.get(state_id)

    def all(self) -> list[State]:
        return self.adapter.all()

    def update(self, state: State) -> None:
        self.adapter.update(state)

    def remove(self, state_id: str) -> None:
        self.adapter.remove(state_id)

    def forget_expired(self, now: int) -> list[State]:
        return self.adapter.forget_expired(now)

    def maybe_compact(self, now: int) -> Optional[State]:
        if isinstance(self.adapter, SummaryMemoryAdapter):
            return self.adapter.compact(now)
        return None

    def lineage(self, state_id: str) -> list[State]:
        """Walk parent_state_id chain back to the root (lineage tracing)."""
        chain: list[State] = []
        cur = self.get(state_id)
        seen = set()
        while cur and cur.state_id not in seen:
            seen.add(cur.state_id)
            chain.append(cur)
            cur = self.get(cur.parent_state_id) if cur.parent_state_id else None
        return chain
