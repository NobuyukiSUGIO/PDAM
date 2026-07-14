"""Common memory-adapter API (M2 Memory Adapter, §11.1).

Three memory方式 are provided (vector / summary / KV, §7.2) behind one
interface so scenarios can swap them and the retriever code stays unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..schema import State


Filter = Callable[[State], bool]


class MemoryAdapter(ABC):
    """Abstract persistent store over ``State`` objects."""

    kind: str = "base"

    def __init__(self) -> None:
        self._states: dict[str, State] = {}

    # -- write side -------------------------------------------------------- #
    @abstractmethod
    def add(self, state: State) -> None:
        ...

    def update(self, state: State) -> None:
        self._states[state.state_id] = state

    def remove(self, state_id: str) -> None:
        self._states.pop(state_id, None)

    # -- read side --------------------------------------------------------- #
    def get(self, state_id: str) -> Optional[State]:
        return self._states.get(state_id)

    def all(self) -> list[State]:
        return list(self._states.values())

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        now: int = 0,
        filter_fn: Optional[Filter] = None,
    ) -> list[tuple[State, float]]:
        """Return up to ``top_k`` (state, score) pairs, best first.

        Expired states (``created_at``/``expires_at``) are never returned.
        """

    # -- maintenance ------------------------------------------------------- #
    def forget_expired(self, now: int) -> list[State]:
        dead = [s for s in self._states.values() if s.expired(now)]
        for s in dead:
            self.remove(s.state_id)
        return dead

    def _candidates(self, now: int, filter_fn: Optional[Filter]) -> list[State]:
        out = []
        for s in self._states.values():
            if s.expired(now):
                continue
            if filter_fn and not filter_fn(s):
                continue
            out.append(s)
        return out
