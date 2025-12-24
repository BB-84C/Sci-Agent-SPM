from __future__ import annotations

import subprocess
import sys
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

    msg = str(action_input.get("message", "Agent failed.")).strip()
    msg_l = msg.lower()
    if any(t in msg_l for t in ["anchor", "roi", "calibrat", "mapping", "not able", "can't set"]):
        ws_path = str(agent.workspace.source_path)
        agent.logger.narrate(f"[Agent] {msg} Launching calibrator to update workspace mappings…")
        cmd = [sys.executable, "-m", "src.calibrate_gui", "--workspace", ws_path]
        try:
            subprocess.Popen(cmd, close_fds=True)
        except Exception:
            pass
        results.append({"action": "launch_calibrator", "action_input": {"workspace": ws_path}, "say": msg})
        agent._emit("result", step=step_index, action="launch_calibrator", result={"workspace": ws_path, "reason": msg})
        return "break"

    raise RuntimeError(msg)
