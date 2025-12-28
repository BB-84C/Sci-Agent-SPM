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
    agent._last_action_signature = None
    msg = str(action_input.get("message", "Agent failed.")).strip()
    raise RuntimeError(msg)
