"""Conversation-summary memory (会話要約型, §7.2).

Holds raw items plus a periodically-compacted summary state. ``compact``
merges the oldest raw items into the running summary — the operation the
A4 要約再構成 attack exploits: benign-looking fragments become one
consolidated instruction inside the summary (§6.4).
"""
from __future__ import annotations

from typing import Optional

from ..embedding import cosine, vectorize
from ..schema import Provenance, State, StateType, TrustLevel, new_id
from .base import Filter, MemoryAdapter


class SummaryMemoryAdapter(MemoryAdapter):
    kind = "summary"

    def __init__(self, compact_threshold: int = 6) -> None:
        super().__init__()
        self.compact_threshold = compact_threshold
        self.summary_id: Optional[str] = None

    def add(self, state: State) -> None:
        self._states[state.state_id] = state

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
            score = cosine(qv, vectorize(s.content))
            # the running summary is always considered "in context"
            if s.state_type is StateType.SUMMARY:
                score = max(score, 0.15)
            if score > 0:
                scored.append((s, score))
        scored.sort(key=lambda p: (p[1], p[0].created_at), reverse=True)
        return scored[:top_k]

    def compact(self, now: int) -> Optional[State]:
        """Merge oldest raw conversation items into the summary state.

        Returns the (new or updated) summary state, or ``None`` if there is
        nothing to compact. Any directive-bearing fragment folded into the
        summary is re-expressed as a single consolidated instruction, and its
        trust is lifted to the summary's own level (信頼度昇格の副作用).
        """
        raw = sorted(
            (s for s in self._states.values()
             if s.state_type is StateType.CONVERSATION and not s.expired(now)),
            key=lambda s: s.created_at,
        )
        if len(raw) < self.compact_threshold:
            return None

        victims = raw[: self.compact_threshold]
        parts = [v.content for v in victims]
        merged_directive = None
        transforms = ["summarize"]
        for v in victims:
            if v.directive is not None:
                # fragments reconstruct into one directive inside the summary
                merged_directive = v.directive
                transforms.append("reconstruct")

        summary_text = "SUMMARY of prior conversation: " + " | ".join(parts)
        if self.summary_id and self.summary_id in self._states:
            summ = self._states[self.summary_id]
            summ.content = summ.content + " || " + summary_text
            summ.provenance = summ.provenance.with_transform("summarize")
            summ.created_at = now
        else:
            summ = State(
                content=summary_text,
                state_type=StateType.SUMMARY,
                provenance=Provenance(author="summarizer", source="internal",
                                      transforms=transforms),
                trust_level=TrustLevel.MEDIUM,
                created_at=now,
                state_id=new_id("sum"),
            )
            self.summary_id = summ.state_id
            self._states[summ.state_id] = summ

        if merged_directive is not None:
            summ.directive = merged_directive
            summ.attack_marker = True  # ground truth: attack survived summarization
            summ.trigger_condition = {"type": merged_directive.trigger.get("type")}
            # record that this state's content/trust is a derived reconstruction,
            # so provenance-aware revalidation can flag it (§10.2).
            if "reconstruct" not in summ.provenance.transforms:
                summ.provenance = summ.provenance.with_transform("reconstruct")

        for v in victims:
            self.remove(v.state_id)
        return summ
