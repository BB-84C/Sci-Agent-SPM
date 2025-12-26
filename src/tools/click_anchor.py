from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..agent import VisualAutomationAgent


def _is_readout_roi(agent: "VisualAutomationAgent", name: str) -> bool:
    try:
        roi = agent.workspace.roi(name)
    except Exception:
        return False
    n = (roi.name or "").lower()
    d = (roi.description or "").lower()
    tags = [str(t).lower() for t in (roi.tags or ())]
    return ("readout" in n) or ("readout" in d) or ("readout" in tags)


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

    # Update last parsed readouts opportunistically from any readout ROIs we just captured.
    try:
        readout_items = [(n, d, img) for (n, d, img) in images if _is_readout_roi(agent, n)]
        if readout_items:
            extracted = agent.llm_tool.extract_readouts(roi_items=[(n, d, img) for (n, d, img) in readout_items])
            agent._accumulate_last_usage(agent.llm_tool)
            vals = extracted.get("values", {})
            unread = extracted.get("unreadable", [])
            readouts: dict[str, str] = {}
            if isinstance(vals, dict):
                for k, v in vals.items():
                    ks = str(k).strip()
                    if ks and v is not None:
                        readouts[ks] = str(v).strip()
            merged = dict(getattr(agent, "_last_readouts", {}) or {})
            merged.update({k: v for k, v in readouts.items() if k and v})
            agent._last_readouts = merged
            if isinstance(unread, list):
                agent._last_unreadable_readouts = [str(x).strip() for x in unread if str(x).strip()]
    except Exception:
        pass
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
