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
    name = str(action_input.get("anchor", "")).strip()
    if not name:
        raise ValueError("click_anchor requires action_input.anchor")
    anchor = agent.workspace.anchor(name)
    step = agent.logger.start_step(f"click_{name}")
    agent.logger.narrate(f"[Click] Clicking anchor {name!r}.")
    agent.actor.click(anchor)
    step.write_meta({"action": "click_anchor", "anchor": name, "say": say})

    agent._last_action_log = f"click_anchor(anchor={name})"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "click_anchor", "action_input": dict(action_input), "say": say})
    agent._emit(
        "result",
        step=step_index,
        action="click_anchor",
        result={"anchor": name, "log_root": str(agent.logger.run_root)},
    )
    return "continue"
