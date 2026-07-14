"""LLM Planner (§6.1 P, §7.1): builds an action plan from retrieved context and current input.

``RuleBasedPlanner`` is a deterministic, reproducible stand-in for a
susceptible LLM. It performs the two operations the threat model cares about:

  1. State synthesis: it reconstructs attack directives from the
     retrieved states, including reassembling A3 split fragments and reading
     directives that survived A4 summarization.
  2. Activation: it evaluates each directive's trigger predicate
     g(x_t, C_t, s_t) (§6.1) against the current context and only emits the
     malicious tool call when the trigger fires — modelling delayed / conditional
     activation rather than immediate injection.

A real LLM can be substituted by implementing ``Planner.plan`` to parse the
natural-language ``State.content`` instead of the structured directive.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .retriever import Retrieved
from .schema import Directive, State, ToolCall, TrustLevel


# --------------------------------------------------------------------------- #
# Plan representation
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    kind: str                       # "benign" | "attack" | "noop"
    text: str = ""
    tool_call: Optional[ToolCall] = None
    directive: Optional[Directive] = None
    inducing_state: Optional[State] = None


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class TriggerContext:
    now: int
    query: str
    task_tags: set[str]
    session_id: str
    injected_session: dict[str, str]   # fragment_group/state_id -> origin session


# --------------------------------------------------------------------------- #
# Trigger evaluation (§6.1 activation predicate g)
# --------------------------------------------------------------------------- #
def trigger_fires(directive: Directive, carrying: State, ctx: TriggerContext) -> bool:
    trig = directive.trigger or {"type": "none"}
    ttype = trig.get("type", "none")

    if ttype == "none":
        return True
    if ttype == "time":
        # absolute tick or elapsed-since-injection
        if "at" in trig:
            return ctx.now >= int(trig["at"])
        if "after" in trig:
            return (ctx.now - carrying.created_at) >= int(trig["after"])
        return True
    if ttype == "event":
        needle = str(trig.get("match", "")).lower()
        hay = ctx.query.lower()
        return needle in hay or needle in {t.lower() for t in ctx.task_tags}
    if ttype == "trust":
        need = TrustLevel.parse(trig.get("min", "medium"))
        return carrying.trust_level >= need
    if ttype == "cross_session":
        origin = ctx.injected_session.get(carrying.state_id) \
            or ctx.injected_session.get(directive.fragment_group or "")
        return origin is not None and origin != ctx.session_id
    return False


# --------------------------------------------------------------------------- #
# Planner interface
# --------------------------------------------------------------------------- #
class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        query: str,
        retrieved: list[Retrieved],
        ctx: TriggerContext,
        benign_action: Optional[ToolCall] = None,
    ) -> Plan:
        ...


class RuleBasedPlanner(Planner):
    """Deterministic susceptible planner (default; runs offline)."""

    def plan(self, query, retrieved, ctx, benign_action=None):
        plan = Plan()

        # (1) benign task action, if the scenario specified one
        if benign_action is not None:
            plan.actions.append(Action(kind="benign", tool_call=benign_action,
                                       text=f"handle task: {query}"))

        # (2) reconstruct directives from retrieved states (state synthesis)
        for directive, carrier in self._reconstruct(retrieved):
            if trigger_fires(directive, carrier, ctx):
                call = ToolCall(
                    tool=directive.tool,
                    op=directive.op,
                    args=dict(directive.args),
                    origin_state_id=carrier.state_id,
                )
                plan.actions.append(Action(
                    kind="attack", tool_call=call,
                    directive=directive, inducing_state=carrier,
                    text=f"reconstructed intent: {directive.goal}",
                ))
                plan.reasoning += f"[fire {directive.goal}] "
        if not plan.actions:
            plan.actions.append(Action(kind="noop", text="no action"))
        return plan

    # -- fragment reassembly (A3) + standalone directives ----------------- #
    def _reconstruct(self, retrieved: list[Retrieved]) -> list[tuple[Directive, State]]:
        out: list[tuple[Directive, State]] = []
        groups: dict[str, list[tuple[Directive, State]]] = defaultdict(list)

        for r in retrieved:
            d = r.state.directive
            if d is None:
                continue
            if d.is_fragment() and d.fragment_group:
                groups[d.fragment_group].append((d, r.state))
            else:
                out.append((d, r.state))

        for group, frags in groups.items():
            total = frags[0][0].fragment_total
            have = {d.fragment_index for d, _ in frags}
            if len(have) < total:
                continue  # incomplete: no reconstruction, each part is benign
            ordered = sorted(frags, key=lambda p: p[0].fragment_index)
            base_d, base_state = ordered[0]
            combined_arg = "".join(d.fragment_part for d, _ in ordered)
            merged = Directive(
                goal=base_d.goal,
                tool=base_d.tool,
                op=base_d.op,
                args={**base_d.args, base_d.args.get("_reassemble_key", "body"): combined_arg}
                if base_d.args.get("_reassemble_key") else {**base_d.args, "body": combined_arg},
                category=base_d.category,
                trigger=base_d.trigger,
                suppress_log=base_d.suppress_log,
                resave=base_d.resave,
            )
            # carrier is the most-trusted fragment (drives revalidation checks)
            carrier = max((s for _, s in ordered), key=lambda s: s.trust_level)
            out.append((merged, carrier))
        return out
