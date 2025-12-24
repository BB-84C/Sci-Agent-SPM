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
    name = str(action_input.get("anchor", "")).strip()
    if not name:
        raise ValueError("click_anchor requires action_input.anchor")
    anchor = agent.workspace.anchor(name)
    step = agent.logger.start_step(f"click_{name}")
    agent.logger.narrate(f"[Click] Clicking anchor {name!r}.")
    agent.actor.click(anchor)

    roi_names = action_input.get("rois", None)
    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
        roi_names = [r.name for r in agent.workspace.rois]
    images = agent._observe_images(roi_names)
    for roi_name, _desc, img in images:
        step.save_image(f"after_{roi_name}.png", img)
    step.write_meta({"action": "click_anchor", "anchor": name, "rois": roi_names, "say": say})

    agent._last_action_log = f"click_anchor(anchor={name})"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "click_anchor", "action_input": dict(action_input), "say": say})
    agent._emit(
        "result",
        step=step_index,
        action="click_anchor",
        result={"anchor": name, "rois": roi_names, "log_root": str(agent.logger.run_root)},
    )
    return "continue"
