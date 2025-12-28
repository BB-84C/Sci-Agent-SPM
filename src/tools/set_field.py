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
    name = str(action_input.get("anchor", "")).strip()
    typed_text = str(action_input.get("typed_text", "")).strip()
    submit = action_input.get("submit", "enter")
    submit_key = None if submit is None else str(submit).strip().lower()
    if not name or not typed_text:
        raise ValueError("set_field requires action_input.anchor and action_input.typed_text")

    anchor = agent.workspace.anchor(name)
    step = agent.logger.start_step(f"set_field_{name}")
    agent.logger.narrate(f"[SetField] Setting {name!r} by typing {typed_text!r}.")

    _clear_focused_field(agent, anchor)
    agent.actor.type_text(typed_text)
    if submit_key in {"enter", "return"}:
        agent.actor.press("enter")
    elif submit_key in {"tab"}:
        agent.actor.press("tab")

    step.write_meta(
        {
            "action": "set_field",
            "anchor": name,
            "typed_text": typed_text,
            "submit": submit_key,
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
        result={"anchor": name, "typed_text": typed_text, "log_root": str(agent.logger.run_root)},
    )
    return "continue"
