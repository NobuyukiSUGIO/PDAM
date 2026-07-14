"""Vector-DB style memory: embedding search over all stored states."""
from __future__ import annotations

from typing import Optional

from ..embedding import cosine, vectorize
from ..schema import State
from .base import Filter, MemoryAdapter


class VectorMemoryAdapter(MemoryAdapter):
    kind = "vector"

    def __init__(self) -> None:
        super().__init__()
        self._vecs: dict[str, object] = {}

    def add(self, state: State) -> None:
        self._states[state.state_id] = state
        self._vecs[state.state_id] = vectorize(state.content)

    def update(self, state: State) -> None:
        super().update(state)
        self._vecs[state.state_id] = vectorize(state.content)

    def remove(self, state_id: str) -> None:
        super().remove(state_id)
        self._vecs.pop(state_id, None)

    def search(
        self,
        query: str,
        top_k: int = 5,
        now: int = 0,
        filter_fn: Optional[Filter] = None,
    ) -> list[tuple[State, float]]:
        qv = vectorize(query)
        scored: list[tuple[State, float]] = []
        for s in self._candidates(now, filter_fn):
            score = cosine(qv, self._vecs[s.state_id])
            if score > 0:
                scored.append((s, score))
        scored.sort(key=lambda p: (p[1], p[0].created_at), reverse=True)
        return scored[:top_k]
