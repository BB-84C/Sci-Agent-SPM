from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..agent import VisualAutomationAgent


def handle(
    agent: "VisualAutomationAgent",
    *,
    step_index: int,
    action_input: Mapping[str, Any],
    say: str,
    signature: Optional[str],
    results: list[Mapping[str, Any]],
) -> Literal["continue", "break"]:
    agent._consecutive_observes = 0
    agent._last_action_signature = None
    agent._observed_since_last_action = True
    results.append({"action": "finish", "action_input": dict(action_input), "say": say})
    # Do not emit a user-facing DONE block here; the agent orchestrator emits a run summary at the end.
    return "break"
