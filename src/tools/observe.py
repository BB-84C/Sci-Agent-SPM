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
    agent._consecutive_observes += 1
    roi_names = action_input.get("rois", None)
    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
        roi_names = [r.name for r in agent.workspace.rois]
    images = agent._observe_images(roi_names)
    agent._last_action_log = f"observe(rois={roi_names})"
    agent._observed_since_last_action = True
    agent._log_observation(
        step_name=f"agent_observe_{step_index}",
        images=images,
        meta={"action": "observe", "action_input": action_input, "say": say},
    )
    results.append({"action": "observe", "action_input": dict(action_input), "say": say})
    agent._emit(
        "result",
        step=step_index,
        action="observe",
        result={"rois": roi_names, "log_root": str(agent.logger.run_root)},
    )
    if agent._consecutive_observes >= 3:
        agent.logger.narrate("[Agent] Observed repeatedly; stopping to avoid loops.")
        results.append({"action": "finish", "action_input": {}, "say": "Done observing."})
        agent._emit("finish", step=step_index, say="Done observing.")
        return "break"
    return "continue"
