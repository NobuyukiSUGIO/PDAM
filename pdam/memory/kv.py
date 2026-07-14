"""Key-value memory (キー・バリュー型, §7.2).

States are keyed by their tags (falling back to salient tokens). Retrieval
matches query tokens against keys, then breaks ties by embedding similarity.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from ..embedding import cosine, tokenize, vectorize
from ..schema import State
from .base import Filter, MemoryAdapter


class KVMemoryAdapter(MemoryAdapter):
    kind = "kv"

    def _keys(self, state: State) -> set[str]:
        if state.tags:
            return {t.lower() for t in state.tags}
        toks = [t for t in tokenize(state.content) if "_" not in t]
        return {t for t, _ in Counter(toks).most_common(4)}

    def add(self, state: State) -> None:
        self._states[state.state_id] = state

    def search(
        self,
        query: str,
        top_k: int = 5,
        now: int = 0,
        filter_fn: Optional[Filter] = None,
    ) -> list[tuple[State, float]]:
        qtokens = set(tokenize(query))
        qv = vectorize(query)
        scored: list[tuple[State, float]] = []
        for s in self._candidates(now, filter_fn):
            keys = self._keys(s)
            overlap = len(keys & qtokens)
            if overlap == 0:
                continue
            score = overlap + cosine(qv, vectorize(s.content))
            scored.append((s, score))
        scored.sort(key=lambda p: (p[1], p[0].created_at), reverse=True)
        return scored[:top_k]
