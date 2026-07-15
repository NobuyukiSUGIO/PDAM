"""Core data schema for the PDAM testbed.

Implements the persistent-state data model of the design document
(Appendix B: data schema). Every persisted item is a ``State`` carrying
provenance, a trust level, a logical creation time, an optional expiry, and
(for injected malicious states) a ground-truth ``directive`` used to compute
attack success mechanically (§9.2 / §12: mechanical judgement).

The module is pure standard library so the whole testbed runs offline.
"""
from __future__ import annotations

import enum
import hashlib
import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def det_unit(*parts) -> float:
    """Deterministic pseudo-random value in [0,1) from the given parts.

    Uses SHA-1 (not Python's salted ``hash``) so results are byte-identical
    across processes and machines — used for reproducible provenance-noise
    injection (§4.2 non-oracle evaluation) and any other seeded jitter."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class StateType(str, enum.Enum):
    """state_type field (Appendix B)."""

    CONVERSATION = "conversation"
    SUMMARY = "summary"
    PROFILE = "profile"
    DOCUMENT = "document"
    TOOL_RESULT = "tool_result"
    PLAN = "plan"
    LOG = "log"


class TrustLevel(enum.IntEnum):
    """trust_level ordinal (untrusted < low < medium < high)."""

    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def parse(cls, value: "TrustLevel | int | str") -> "TrustLevel":
        if isinstance(value, TrustLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper()]


class PolicyViolation(str, enum.Enum):
    """policy_violation enum (Appendix B / §4.2)."""

    NONE = "none"
    EXFILTRATION = "exfiltration"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    INTEGRITY = "integrity"
    TRACE_SUPPRESSION = "trace_suppression"


class AttackType(str, enum.Enum):
    """Attack taxonomy A1-A8 (§5.1)."""

    A1_TIME_DELAY = "A1"          # time-delayed
    A2_EVENT_CONDITIONAL = "A2"   # event-conditional
    A3_SPLIT_STATE = "A3"         # split-state
    A4_SUMMARY_RECONSTRUCT = "A4" # summary-reconstruction
    A5_TRUST_ESCALATION = "A5"    # trust-escalation
    A6_CROSS_SESSION = "A6"       # cross-session
    A7_SELF_RESAVE = "A7"         # self-re-save
    A8_TRACE_SUPPRESSION = "A8"   # trace-suppression


ATTACK_NAMES = {
    AttackType.A1_TIME_DELAY: "time-delayed",
    AttackType.A2_EVENT_CONDITIONAL: "event-conditional",
    AttackType.A3_SPLIT_STATE: "split-state",
    AttackType.A4_SUMMARY_RECONSTRUCT: "summary-reconstruction",
    AttackType.A5_TRUST_ESCALATION: "trust-escalation",
    AttackType.A6_CROSS_SESSION: "cross-session",
    AttackType.A7_SELF_RESAVE: "self-re-save",
    AttackType.A8_TRACE_SUPPRESSION: "trace-suppression",
}


# --------------------------------------------------------------------------- #
# Identifier helper
# --------------------------------------------------------------------------- #
_counter = itertools.count(1)


def new_id(prefix: str = "st") -> str:
    """Deterministic monotonically-increasing id (reproducible, no RNG)."""
    return f"{prefix}_{next(_counter):06d}"


def reset_ids() -> None:
    """Reset the id counter so repeated runs produce identical ids."""
    global _counter
    _counter = itertools.count(1)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@dataclass
class Provenance:
    """Provenance: producing agent, source, transform history (§3.2)."""

    author: str = "system"           # who produced the state
    source: str = "internal"         # where it was obtained (email/doc/web/...)
    transforms: list[str] = field(default_factory=list)  # transform history (summarize/re-save/...)
    external: bool = False           # obtained from an attacker-controllable surface

    def with_transform(self, name: str) -> "Provenance":
        return Provenance(
            author=self.author,
            source=self.source,
            transforms=[*self.transforms, name],
            external=self.external,
        )


# --------------------------------------------------------------------------- #
# Directive  (ground-truth attack payload)
# --------------------------------------------------------------------------- #
@dataclass
class Directive:
    """The malicious intent carried by an injected state.

    The rule-based planner (a deterministic stand-in for a susceptible LLM)
    reconstructs this directive from retrieved states and, when the trigger is
    satisfied, emits the corresponding tool call. A real LLM planner would
    instead parse ``State.content``; the structured directive keeps the
    simulation reproducible while modelling the same causal chain.
    """

    goal: str                                  # human-readable attack goal
    tool: str                                  # target tool (e.g. "email")
    op: str                                    # operation (e.g. "send")
    args: dict[str, Any] = field(default_factory=dict)
    category: PolicyViolation = PolicyViolation.UNAUTHORIZED_ACTION
    trigger: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    # A3 split-state fragmentation
    fragment_group: Optional[str] = None
    fragment_index: int = 0
    fragment_total: int = 1
    fragment_part: str = ""                    # partial argument payload
    # A8 trace suppression
    suppress_log: bool = False
    # A7 self-re-save
    resave: bool = False

    def is_fragment(self) -> bool:
        return self.fragment_total > 1


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class State:
    """A single persisted memory item (Appendix B schema)."""

    content: str
    state_type: StateType = StateType.CONVERSATION
    session_id: str = "s0"
    run_id: str = "r0"
    provenance: Provenance = field(default_factory=Provenance)
    trust_level: TrustLevel = TrustLevel.MEDIUM
    created_at: int = 0                        # logical tick
    expires_at: Optional[int] = None           # logical tick of forgetting
    retrieval_score: float = 0.0               # last similarity score
    trigger_condition: dict[str, Any] = field(default_factory=dict)
    state_id: str = field(default_factory=lambda: new_id("st"))
    parent_state_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    # ground-truth annotations (used only by the evaluator / monitor)
    attack_marker: bool = False                # True => injected by attacker
    directive: Optional[Directive] = None

    def expired(self, now: int) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state_type"] = self.state_type.value
        d["trust_level"] = int(self.trust_level)
        return d


# --------------------------------------------------------------------------- #
# Tool call and log event
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """tool_call object (Appendix B): tool name, arguments, decision, result."""

    tool: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"          # allow / deny / approve
    mediated: bool = True
    result: Any = None
    origin_state_id: Optional[str] = None  # state that induced this call (lineage)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogEvent:
    """Time-ordered audit event (§11.2). Carries the full id chain."""

    event_type: str
    tick: int
    run_id: str
    session_id: str
    task_id: str
    state_id: Optional[str] = None
    parent_state_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
