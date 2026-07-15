"""Real-framework memory adapters backed by LangChain + Chroma (§5.4).

These implement the same ``MemoryAdapter`` contract as the synthetic backends,
but delegate embedding, similarity search, TTL, and (for the summary backend)
summarisation to real components:

    * embedding + ANN retrieval : LangChain ``Chroma`` over LM Studio embeddings
    * summarisation (A4)        : a real LLM summarise call (LM Studio chat)

The full ``State`` object stays in ``self._states`` as the reconstruction map
(so the mechanical judge and lifecycle funnel keep their ground-truth labels),
while Chroma does the real write/retrieve work keyed by ``state_id``. The
DEFENSE only ever reads the provenance a real framework actually exposes, so
lineage lost through real summarisation is *measured*, not assumed (§5.3/§5.8).

Requires the ``.venv-real`` extras (see ``requirements-real.txt``); importing
this module without them raises a clear error, leaving the pure-stdlib testbed
unaffected.
"""
from __future__ import annotations

import os
from typing import Optional

from ..schema import (
    Provenance,
    State,
    StateType,
    TrustLevel,
    new_id,
)
from .base import Filter, MemoryAdapter

_LMSTUDIO_BASE = os.environ.get("PDAM_LMSTUDIO_BASE", "http://localhost:1234/v1")
_EMBED_MODEL = os.environ.get("PDAM_EMBED_MODEL",
                              "text-embedding-nomic-embed-text-v1.5")
_SUMM_MODEL = os.environ.get("PDAM_SUMM_MODEL", "qwen/qwen3.5-9b")

_EMBEDDING = None  # module-level singleton so we build one HTTP client, not one/run


def _embedding():
    global _EMBEDDING
    if _EMBEDDING is None:
        from langchain_openai import OpenAIEmbeddings
        _EMBEDDING = OpenAIEmbeddings(
            model=_EMBED_MODEL, base_url=_LMSTUDIO_BASE, api_key="lm-studio",
            check_embedding_ctx_length=False,
        )
    return _EMBEDDING


def _new_chroma():
    """A fresh, isolated in-memory Chroma collection for one run.

    Chroma's in-process client shares a backend across instances, so the
    collection name must be process-globally unique (``new_id`` resets per run
    via ``reset_ids`` and would collide)."""
    import uuid

    import chromadb
    from langchain_chroma import Chroma
    client = chromadb.EphemeralClient()
    return Chroma(
        client=client,
        collection_name="pdam_" + uuid.uuid4().hex,
        embedding_function=_embedding(),
    )


def _to_document(state: State):
    from langchain_core.documents import Document
    # Chroma metadata must be scalar; we only need state_id to map back, plus a
    # couple of scalars for real-store filtering (TTL / trust). Ground-truth
    # labels stay in self._states, never in the store the defense could read.
    md = {
        "state_id": state.state_id,
        "created_at": int(state.created_at),
        "trust": int(state.trust_level),
    }
    if state.expires_at is not None:
        md["expires_at"] = int(state.expires_at)
    return Document(page_content=state.content or " ", metadata=md, id=state.state_id)


class LCVectorAdapter(MemoryAdapter):
    """LangChain + Chroma vector store with real LM Studio embeddings."""

    kind = "lc_vector"

    def __init__(self) -> None:
        super().__init__()
        self._chroma = _new_chroma()

    # -- write ------------------------------------------------------------- #
    def add(self, state: State) -> None:
        self._states[state.state_id] = state
        self._chroma.add_documents([_to_document(state)], ids=[state.state_id])

    def remove(self, state_id: str) -> None:
        super().remove(state_id)
        try:
            self._chroma.delete(ids=[state_id])
        except Exception:
            pass

    def forget_expired(self, now: int) -> list[State]:
        dead = super().forget_expired(now)
        if dead:
            try:
                self._chroma.delete(ids=[s.state_id for s in dead])
            except Exception:
                pass
        return dead

    # -- read -------------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = 5,
        now: int = 0,
        filter_fn: Optional[Filter] = None,
    ) -> list[tuple[State, float]]:
        if not self._states:
            return []
        # real ANN retrieval; over-fetch so TTL / filter drops still leave top_k
        hits = self._chroma.similarity_search_with_score(query, k=top_k * 3)
        out: list[tuple[State, float]] = []
        for doc, distance in hits:
            sid = doc.metadata.get("state_id")
            st = self._states.get(sid)
            if st is None or st.expired(now):
                continue
            if filter_fn and not filter_fn(st):
                continue
            out.append((st, 1.0 / (1.0 + float(distance))))  # monotone in distance
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:top_k]


class LCSummaryAdapter(LCVectorAdapter):
    """LangChain vector store whose old conversations are periodically compacted
    by a real LLM summariser (the substrate for A4). Summarisation is where real
    lineage is lost: the summary node carries only ``summarize`` provenance, so a
    defense that trusts it is laundering-blind exactly as in deployment."""

    kind = "lc_summary"
    COMPACT_EVERY = 4          # ticks between compactions
    _last_compact = -1

    def compact(self, now: int) -> Optional[State]:
        if now - self._last_compact < self.COMPACT_EVERY:
            return None
        self._last_compact = now
        # compact conversation + document states older than this window
        victims = [s for s in self._states.values()
                   if s.state_type in (StateType.CONVERSATION, StateType.DOCUMENT)
                   and not s.expired(now)]
        if len(victims) < 2:
            return None
        text = "\n".join(f"- {s.content}" for s in victims)
        summary_text = _summarise(text)
        # a real summary carries the reconstructed intent if the fragments were
        # present; ground-truth attack_marker/directive propagate for JUDGING,
        # but provenance is the laundered (agent-authored, summarize) view the
        # defense sees.
        carried = [s for s in victims if s.attack_marker]
        directive = carried[0].directive if carried else None
        summ = State(
            content=summary_text,
            state_type=StateType.SUMMARY,
            session_id=victims[-1].session_id,
            provenance=Provenance(author="agent", source="summary",
                                  transforms=["summarize"], external=False),
            trust_level=TrustLevel.MEDIUM,
            created_at=now,
            parent_state_id=carried[0].state_id if carried else None,
            directive=directive,
            attack_marker=bool(carried),
            tags=list(victims[-1].tags),
        )
        for s in victims:
            self.remove(s.state_id)
        self.add(summ)
        return summ


class LCKVAdapter(MemoryAdapter):
    """Key-value backend (tag/topic keyed) with no embedding retrieval — the
    third backend the synthetic matrix never exercised (reviewer §5.4/#6). Keys
    are the state's tags; ``search`` matches query tokens against keys."""

    kind = "lc_kv"

    def add(self, state: State) -> None:
        self._states[state.state_id] = state

    def search(
        self,
        query: str,
        top_k: int = 5,
        now: int = 0,
        filter_fn: Optional[Filter] = None,
    ) -> list[tuple[State, float]]:
        q = set(query.lower().split())
        out: list[tuple[State, float]] = []
        for st in self._candidates(now, filter_fn):
            keys = set(t.lower() for t in st.tags)
            overlap = len(q & keys)
            if overlap:
                out.append((st, float(overlap)))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:top_k]


def _summarise(text: str) -> str:
    """Call the LM Studio chat model to actually summarise; fall back to a
    concatenation if the server is unreachable (keeps offline tests runnable)."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=_LMSTUDIO_BASE, api_key="lm-studio")
        r = client.chat.completions.create(
            model=_SUMM_MODEL, temperature=0.0, max_tokens=200,
            messages=[
                {"role": "system", "content":
                 "Summarise the following memory notes into one concise paragraph, "
                 "preserving any concrete instructions or preferences."},
                {"role": "user", "content": text},
            ],
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "Summary of prior notes: " + " ".join(text.split())[:400]
