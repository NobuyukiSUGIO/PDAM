"""Logger / Monitor event log (§7.1, §11.2).

Every event carries run_id / session_id / task_id / state_id / parent_state_id
and a logical tick, and is stored in time order so tables and figures can be
regenerated deterministically (§12.1 再現性).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .schema import LogEvent


@dataclass
class EventLog:
    run_id: str
    events: list[LogEvent] = field(default_factory=list)

    def emit(self, event_type: str, tick: int, session_id: str, task_id: str,
             state_id=None, parent_state_id=None, **payload) -> LogEvent:
        ev = LogEvent(
            event_type=event_type, tick=tick, run_id=self.run_id,
            session_id=session_id, task_id=task_id,
            state_id=state_id, parent_state_id=parent_state_id, payload=payload,
        )
        self.events.append(ev)
        return ev

    def of_type(self, event_type: str) -> list[LogEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self.events)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_jsonl())
