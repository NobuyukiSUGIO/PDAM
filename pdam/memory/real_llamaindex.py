"""Real-framework memory adapters backed by LlamaIndex + Chroma (§5.4).

Mirror of ``real_langchain`` for the second framework required by the review
(§5.4, R2). Same contract, same ``self._states`` reconstruction map; embedding
and retrieval are performed by a LlamaIndex ``VectorStoreIndex`` over a real
Chroma vector store with LM Studio embeddings, and A4 compaction uses a real LLM.
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
from .real_langchain import LCKVAdapter, _summarise  # KV needs no embeddings; reuse

_LMSTUDIO_BASE = os.environ.get("PDAM_LMSTUDIO_BASE", "http://localhost:1234/v1")
_EMBED_MODEL = os.environ.get("PDAM_EMBED_MODEL",
                              "text-embedding-nomic-embed-text-v1.5")

_EMBED = None


def _embed_model():
    global _EMBED
    if _EMBED is None:
        from llama_index.embeddings.openai import OpenAIEmbedding
        _EMBED = OpenAIEmbedding(
            model_name=_EMBED_MODEL, api_base=_LMSTUDIO_BASE, api_key="lm-studio",
        )
    return _EMBED


class LIVectorAdapter(MemoryAdapter):
    """LlamaIndex ``VectorStoreIndex`` + Chroma with real LM Studio embeddings."""

    kind = "li_vector"

    def __init__(self) -> None:
        super().__init__()
        import uuid

        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = chromadb.EphemeralClient()
        collection = client.create_collection("pdam_" + uuid.uuid4().hex)
        vstore = ChromaVectorStore(chroma_collection=collection)
        storage = StorageContext.from_defaults(vector_store=vstore)
        self._index = VectorStoreIndex(
            nodes=[], storage_context=storage, embed_model=_embed_model(),
        )

    def _node(self, state: State):
        from llama_index.core.schema import TextNode
        return TextNode(
            text=state.content or " ", id_=state.state_id,
            metadata={"state_id": state.state_id,
                      "created_at": int(state.created_at),
                      "trust": int(state.trust_level)},
        )

    # -- write ------------------------------------------------------------- #
    def add(self, state: State) -> None:
        self._states[state.state_id] = state
        self._index.insert_nodes([self._node(state)])

    def remove(self, state_id: str) -> None:
        super().remove(state_id)
        try:
            self._index.delete_nodes([state_id])
        except Exception:
            pass

    def forget_expired(self, now: int) -> list[State]:
        dead = super().forget_expired(now)
        for s in dead:
            try:
                self._index.delete_nodes([s.state_id])
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
        retriever = self._index.as_retriever(similarity_top_k=top_k * 3)
        hits = retriever.retrieve(query)
        out: list[tuple[State, float]] = []
        for nws in hits:
            sid = nws.node.metadata.get("state_id")
            st = self._states.get(sid)
            if st is None or st.expired(now):
                continue
            if filter_fn and not filter_fn(st):
                continue
            out.append((st, float(nws.score if nws.score is not None else 0.0)))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:top_k]


class LISummaryAdapter(LIVectorAdapter):
    """LlamaIndex vector store with real-LLM periodic compaction (A4 substrate)."""

    kind = "li_summary"
    COMPACT_EVERY = 4
    _last_compact = -1

    def compact(self, now: int) -> Optional[State]:
        if now - self._last_compact < self.COMPACT_EVERY:
            return None
        self._last_compact = now
        victims = [s for s in self._states.values()
                   if s.state_type in (StateType.CONVERSATION, StateType.DOCUMENT)
                   and not s.expired(now)]
        if len(victims) < 2:
            return None
        text = "\n".join(f"- {s.content}" for s in victims)
        summary_text = _summarise(text)
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


class LIKVAdapter(LCKVAdapter):
    """Key-value backend (no embeddings); identical mechanics to the LangChain
    KV backend since neither uses a vector index."""

    kind = "li_kv"
