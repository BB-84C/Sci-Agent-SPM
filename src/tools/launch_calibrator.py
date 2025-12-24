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
    ws_path = str(agent.workspace.source_path)
    cmd = [sys.executable, "-m", "src.calibrate_gui", "--workspace", ws_path]
    try:
        subprocess.Popen(cmd, close_fds=True)
    except Exception:
        pass
    agent._last_action_log = "launch_calibrator()"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    agent.logger.narrate(f"[Agent] Launched calibrator for {ws_path}. Save and rerun the command.")
    results.append({"action": "launch_calibrator", "action_input": dict(action_input), "say": say})
    agent._emit("result", step=step_index, action="launch_calibrator", result={"workspace": ws_path})
    return "break"
