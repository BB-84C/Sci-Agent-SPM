from __future__ import annotations

import time
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
    seconds_raw = action_input.get("seconds", None)
    if seconds_raw is None:
        raise ValueError("wait requires action_input.seconds")
    if not isinstance(seconds_raw, (int, float)):
        raise ValueError("wait requires numeric action_input.seconds")

    seconds = float(seconds_raw)
    if seconds < 0:
        raise ValueError("wait requires non-negative action_input.seconds")

    agent._consecutive_observes = 0
    step = agent.logger.start_step("wait")
    agent.logger.narrate(f"[Wait] Waiting {seconds:.1f} seconds.")

    meta: dict[str, Any] = {
        "tool": "wait",
        "seconds": seconds,
        "dry_run": agent.dry_run,
    }
    step.write_meta(meta)

    agent._last_action_log = f"wait(seconds={seconds:.1f})"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "wait", "action_input": dict(action_input), "result": meta, "say": say})
    agent._emit("result", step=step_index, action="wait", result=meta | {"log_root": str(agent.logger.run_root)})

    if not agent.dry_run and seconds > 0:
        time.sleep(min(seconds, 6 * 60 * 60))
    return "continue"
