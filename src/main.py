from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .workspace import Workspace, load_workspace

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visual automation agent MVP (screenshots + fixed anchors).")
    ap.add_argument("--agent", action="store_true", help="Start the agent TUI (chat mode).")

    args = ap.parse_args(argv)

    try:
        if not args.agent:
            raise ValueError("Non-agent mode has been removed. Start the TUI with `--agent`.")

        from .agent import AgentConfig, VisualAutomationAgent
        from .chat_tui import run_chat_tui

        ws_path = Path("workspace.json")
        if not ws_path.exists():
            raise ValueError("Missing workspace.json. Create one with `python -m src.calibrate_gui --workspace workspace.json`.")
        workspace: Workspace = load_workspace(ws_path)
        agent = VisualAutomationAgent(
            workspace=workspace,
            config=AgentConfig(),
            dry_run=False,
        )
        return run_chat_tui(agent=agent, first_message=None)
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
