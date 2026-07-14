"""Tool Broker (§7.1): ポリシー確認・権限付与・ツール呼出.

Every tool call is mediated: the broker asks the PolicyMonitor to authorize
the call (execution-time defenses), executes it in the sandbox only if allowed,
and returns the completed ``ToolCall`` plus the sandbox ``ToolEffect``.
"""
from __future__ import annotations

from typing import Optional

from .policy import Decision, PolicyMonitor
from .schema import State, ToolCall
from .tools.sandbox import ToolEffect, ToolSandbox


class ToolBroker:
    def __init__(self, sandbox: ToolSandbox, monitor: PolicyMonitor) -> None:
        self.sandbox = sandbox
        self.monitor = monitor

    def is_high_privilege(self, tool: str, op: str) -> bool:
        return self.sandbox.has(tool) and op in self.sandbox.get(tool).high_privilege_ops

    def execute(
        self,
        call: ToolCall,
        inducing: Optional[State],
        authorized_ops: set[str],
    ) -> tuple[ToolCall, ToolEffect, Decision]:
        if not self.sandbox.has(call.tool):
            call.decision = "deny"
            call.result = {"error": f"no such tool {call.tool}"}
            return call, ToolEffect(), Decision(False, "exec", "unknown_tool")

        tool = self.sandbox.get(call.tool)
        high = call.op in tool.high_privilege_ops

        # dry-run the effect so execution-time defenses can inspect it, without
        # committing side effects until authorized.
        preview, effect = tool.invoke(call.op, call.args)
        # roll back the speculative record; we re-invoke on allow
        if tool.calls:
            tool.calls.pop()

        decision = self.monitor.authorize(
            call, effect, inducing, authorized_ops, high
        )
        call.decision = "allow" if decision.allow else "deny"

        if decision.allow:
            result, effect = tool.invoke(call.op, call.args)
            call.result = result
        else:
            call.result = {"blocked_by": decision.blocked_by, "reason": decision.reason}
            effect = ToolEffect()  # nothing happened

        return call, effect, decision
