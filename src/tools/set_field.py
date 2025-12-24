from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..agent import VisualAutomationAgent


def _clear_focused_field(agent: "VisualAutomationAgent", anchor: Any) -> None:
    actor = agent.actor
    # Robust field clearing: different apps/controls support different selection semantics.
    actor.double_click(anchor)          # focus / often selects a token
    actor.hotkey("ctrl", "a")           # try select-all
    actor.press("delete")               # clear selection
    actor.hotkey("ctrl", "a")           # repeat (some controls need it after focus)
    actor.press("backspace")            # clear selection
    actor.press("home")                 # fallback selection: start…
    actor.hotkey("shift", "end")        # …to end
    actor.press("delete")               # clear
    actor.hotkey("ctrl", "a")           # final sweep
    actor.press("backspace")


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
    typed_text = str(action_input.get("typed_text", "")).strip()
    submit = action_input.get("submit", "enter")
    submit_key = None if submit is None else str(submit).strip().lower()
    roi_names = action_input.get("rois", None)
    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
        roi_names = [r.name for r in agent.workspace.rois]
    if not name or not typed_text:
        raise ValueError("set_field requires action_input.anchor and action_input.typed_text")

    anchor = agent.workspace.anchor(name)
    step = agent.logger.start_step(f"set_field_{name}")
    agent.logger.narrate(f"[SetField] Setting {name!r} by typing {typed_text!r}.")

    before_imgs = agent._observe_images(roi_names)
    for roi_name, _desc, img in before_imgs:
        step.save_image(f"before_{roi_name}.png", img)

    _clear_focused_field(agent, anchor)
    agent.actor.type_text(typed_text)
    if submit_key in {"enter", "return"}:
        agent.actor.press("enter")
    elif submit_key in {"tab"}:
        agent.actor.press("tab")

    after_imgs = agent._observe_images(roi_names)
    for roi_name, _desc, img in after_imgs:
        step.save_image(f"after_{roi_name}.png", img)

    step.write_meta(
        {
            "action": "set_field",
            "anchor": name,
            "typed_text": typed_text,
            "submit": submit_key,
            "rois": roi_names,
            "say": say,
        }
    )
    agent._last_action_log = f"set_field(anchor={name}, typed_text={typed_text!r})"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "set_field", "action_input": dict(action_input), "say": say})
    agent._emit(
        "result",
        step=step_index,
        action="set_field",
        result={"anchor": name, "typed_text": typed_text, "rois": roi_names, "log_root": str(agent.logger.run_root)},
    )
    return "continue"
