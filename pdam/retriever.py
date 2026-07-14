"""Retriever / RAG (§7.1): embedding search, filtering, ranking.

Retrieves candidate states from memory then applies retrieval-time defenses
(embedding/anomaly filter, provenance revalidation, low-trust fragment
co-retrieval limit) via the PolicyMonitor before handing context to the planner.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory.store import MemoryStore
from .policy import PolicyMonitor
from .schema import State


@dataclass
class Retrieved:
    state: State
    score: float


class Retriever:
    def __init__(self, store: MemoryStore, monitor: PolicyMonitor, top_k: int = 5) -> None:
        self.store = store
        self.monitor = monitor
        self.top_k = top_k

    def retrieve(self, query: str, now: int) -> list[Retrieved]:
        raw = self.store.adapter.search(query, top_k=self.top_k * 2, now=now)
        filtered = self.monitor.filter_candidates(raw, now)
        out = []
        for state, score in filtered[: self.top_k]:
            state.retrieval_score = score
            out.append(Retrieved(state, score))
        return out
