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
    agent._consecutive_observes += 1
    roi_names = action_input.get("rois", None)
    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
        roi_names = [r.name for r in agent.workspace.rois]
    images = agent._observe_images(roi_names)

    readout_names = [n for n in roi_names if _is_readout_roi(agent, n)]
    readouts: dict[str, Optional[str]] = {}
    unreadable: list[str] = []
    if readout_names:
        roi_objs = [agent.workspace.roi(n) for n in readout_names]
        roi_descs = [getattr(r, "description", "") or "" for r in roi_objs]
        img_map = {n: img for (n, _d, img) in images}
        readout_items = [(n, roi_descs[readout_names.index(n)], img_map[n]) for n in readout_names if n in img_map]
        try:
            extracted = agent.llm_tool.extract_readouts(roi_items=readout_items)
            agent._accumulate_last_usage(agent.llm_tool)
            vals = extracted.get("values", {})
            if isinstance(vals, dict):
                for k, v in vals.items():
                    ks = str(k).strip()
                    if ks and ks in readout_names:
                        readouts[ks] = None if v is None else str(v).strip()
            unread = extracted.get("unreadable", [])
            if isinstance(unread, list):
                unreadable = [str(x).strip() for x in unread if str(x).strip() and str(x).strip() in readout_names]
        except Exception:
            unreadable = list(readout_names)

        # Emit an explicit assistant-visible report so the user always gets the value (or a clear failure).
        lines = []
        for n in readout_names:
            val = readouts.get(n, None)
            if val:
                lines.append(f"{n}: {val}")
        for n in unreadable:
            if n not in readouts or not readouts.get(n):
                lines.append(
                    f"{n}: (unreadable) — please recalibrate/expand this ROI so the full value and units are visible."
                )
        if lines:
            agent._remember("Readouts:\n" + "\n".join(lines))
    # Persist the last extracted readouts so the next observation text can include them as plain text.
    try:
        agent._last_readouts = {k: v for k, v in readouts.items() if isinstance(k, str) and isinstance(v, str) and v.strip()}
        agent._last_unreadable_readouts = [x for x in unreadable if isinstance(x, str) and x.strip()]
    except Exception:
        pass

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
        result={
            "rois": roi_names,
            "log_root": str(agent.logger.run_root),
            "readouts": {k: v for k, v in readouts.items() if v},
            "unreadable": unreadable,
        },
    )
    pending = getattr(agent, "_pending_confirm_readout_rois", []) or []
    if agent._consecutive_observes >= 3 and not pending:
        agent.logger.narrate("[Agent] Observed repeatedly; stopping to avoid loops.")
        results.append({"action": "finish", "action_input": {}, "say": "Done observing."})
        agent._emit("finish", step=step_index, say="Done observing.")
        return "break"
    return "continue"
