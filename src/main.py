from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .tui_settings import DEFAULT_SETTINGS_PATH, load_tui_settings
from .workspace import Workspace, load_workspace


def _resolve_workspace_path(raw: str, *, repo_root: Path, settings_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    alt = settings_path.parent / path
    if alt.exists():
        return alt
    return path

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visual automation agent MVP (screenshots + fixed anchors).")
    ap.add_argument("--agent", action="store_true", help="Start the agent TUI (chat mode).")

    args = ap.parse_args(argv)

    try:
        if not args.agent:
            raise ValueError("Non-agent mode has been removed. Start the TUI with `--agent`.")

        from .agent import AgentConfig, VisualAutomationAgent
        from .chat_tui import run_chat_tui

        repo_root = Path(__file__).resolve().parent.parent
        settings_path = repo_root / DEFAULT_SETTINGS_PATH
        st = load_tui_settings(settings_path)
        ws_path = repo_root / "workspace.json"
        if st.workspace:
            ws_path = _resolve_workspace_path(st.workspace, repo_root=repo_root, settings_path=settings_path)
        if not ws_path.exists():
            raise ValueError(
                f"Missing workspace file: {ws_path}. Update sessions/.tui_settings.json "
                "or create one with `python -m src.calibrate_gui --workspace workspace.json`."
            )
        workspace: Workspace = load_workspace(ws_path)
        agent = VisualAutomationAgent(
            workspace=workspace,
            config=AgentConfig(),
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
