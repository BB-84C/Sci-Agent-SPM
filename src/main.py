from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional

from .actions import ActionConfig, Actor
from .abort import start_abort_hotkey
from .capture import ScreenCapturer
from .logger import RunLogger
from .skills.set_bias import SetBiasParams
from .skills.start_scan import StartScanParams
from .workspace import Workspace, load_workspace


@dataclass(frozen=True, slots=True)
class PlanStep:
    name: str
    args: Mapping[str, Any]


_SET_BIAS_TO_RE = re.compile(r"\bset\b.*\bbias\b.*\bto\b\s*(-?\d+(?:\.\d+)?)\s*(m?v)\b", re.IGNORECASE)
_SET_BIAS_FROM_TO_RE = re.compile(
    r"\bset\b.*\bbias\b.*\bfrom\b\s*(-?\d+(?:\.\d+)?)\s*(m?v)\b.*\bto\b\s*(-?\d+(?:\.\d+)?)\s*(m?v)\b",
    re.IGNORECASE,
)
_START_SCAN_RE = re.compile(r"\b(start scanning|start scan)\b", re.IGNORECASE)
_SCAN_FROM_TOP_RE = re.compile(r"\b(from top|top)\b", re.IGNORECASE)
_SCAN_FROM_BOTTOM_RE = re.compile(r"\b(from bottom|bottom)\b", re.IGNORECASE)


def parse_command(command: str) -> list[PlanStep]:
    cmd = " ".join(command.strip().split())
    steps: list[PlanStep] = []

    m = _SET_BIAS_FROM_TO_RE.search(cmd)
    if m:
        target_value = float(m.group(3))
        unit = m.group(4)
        unit = "V" if unit.lower() == "v" else "mV"
        steps.append(PlanStep(name="SetBias", args={"target_value": target_value, "target_unit": unit}))
    else:
        m2 = _SET_BIAS_TO_RE.search(cmd)
        if m2:
            target_value = float(m2.group(1))
            unit = m2.group(2)
            unit = "V" if unit.lower() == "v" else "mV"
            steps.append(PlanStep(name="SetBias", args={"target_value": target_value, "target_unit": unit}))

    if _START_SCAN_RE.search(cmd):
        direction = "default"
        if _SCAN_FROM_TOP_RE.search(cmd) and not _SCAN_FROM_BOTTOM_RE.search(cmd):
            direction = "top"
        elif _SCAN_FROM_BOTTOM_RE.search(cmd) and not _SCAN_FROM_TOP_RE.search(cmd):
            direction = "bottom"
        steps.append(PlanStep(name="StartScan", args={"direction": direction}))

    if not steps:
        raise ValueError(
            "Could not parse command. Try: "
            "'set the tip bias to 200 mV', "
            "'set bias from 100 mV to 200 mV', "
            "'start scan'."
        )
    return steps


def _print_plan(steps: Iterable[PlanStep]) -> None:
    print("Plan:")
    for i, s in enumerate(steps, start=1):
        args = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
        print(f"  {i}. {s.name}({args})")


def run_plan(
    *,
    workspace: Workspace,
    steps: List[PlanStep],
    dry_run: bool,
    no_ocr: bool,
    require_ocr: bool,
    abort_hotkey: bool,
    action_delay_s: float,
    max_attempts: int,
    tolerance_mv: float,
    log_dir: str,
) -> list[Mapping[str, Any]]:
    logger = RunLogger(root_dir=log_dir)
    capturer = ScreenCapturer()
    abort = start_abort_hotkey() if abort_hotkey else None
    actor = Actor(dry_run=dry_run, config=ActionConfig(delay_s=action_delay_s), abort_event=abort.event if abort else None)

    logger.narrate(f"Workspace: {workspace.source_path}")
    logger.narrate(f"Dry-run: {dry_run}")
    logger.narrate("Failsafe: move mouse to top-left to abort (pyautogui.FAILSAFE)")
    if abort_hotkey:
        logger.narrate("Abort hotkey: ESC")
    if no_ocr or require_ocr:
        logger.narrate("Note: OCR flags are deprecated; verification is screenshot-based.")

    results: list[Mapping[str, Any]] = []
    try:
        for step in steps:
            if step.name == "SetBias":
                params = SetBiasParams(
                    target_value=float(step.args["target_value"]),
                    target_unit=str(step.args.get("target_unit", "mV")),  # type: ignore[arg-type]
                )
                from .skills.set_bias import run as run_set_bias

                results.append(
                    run_set_bias(
                        workspace=workspace,
                        capturer=capturer,
                        actor=actor,
                        logger=logger,
                        params=params,
                    )
                )
            elif step.name == "StartScan":
                direction = str(step.args.get("direction", "default")).lower()
                if direction not in {"top", "bottom", "default"}:
                    direction = "default"
                params = StartScanParams(direction=direction)  # type: ignore[arg-type]
                from .skills.start_scan import run as run_start_scan

                results.append(
                    run_start_scan(
                        workspace=workspace,
                        capturer=capturer,
                        actor=actor,
                        logger=logger,
                        params=params,
                    )
                )
            else:
                raise ValueError(f"Unknown step: {step.name}")
    finally:
        if abort is not None:
            abort.stop()

    logger.narrate(f"Logs: {logger.run_root}")
    return results


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visual automation agent MVP (screenshots + fixed anchors).")
    ap.add_argument("--workspace", required=True, help="Path to workspace.json")
    ap.add_argument("--command", default="", help="Natural-language command (optional with --chat)")
    ap.add_argument("--agent", action="store_true", help="Use LLM agent mode (multimodal ReAct)")
    ap.add_argument("--chat", action="store_true", help="Interactive chat loop (agent mode only)")
    ap.add_argument("--chat-plain", action="store_true", help="Plain stdin/stdout chat (no TUI)")
    ap.add_argument("--model", default="gpt-5.2", help="OpenAI model name for --agent")
    ap.add_argument("--max-agent-steps", type=int, default=10, help="Max ReAct steps for --agent")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and log screenshots, but do not click/type")
    ap.add_argument("--no-ocr", action="store_true", help="(Deprecated) No-op; verification is screenshot-based")
    ap.add_argument("--require-ocr", action="store_true", help="(Deprecated) No-op; verification is screenshot-based")
    ap.add_argument("--no-abort-hotkey", action="store_true", help="Disable ESC abort hotkey listener")
    ap.add_argument("--action-delay", type=float, default=0.25, help="Delay between atomic UI actions (seconds)")
    ap.add_argument("--max-attempts", type=int, default=3, help="(Deprecated) No-op; retries are agent-driven")
    ap.add_argument("--tolerance-mv", type=float, default=2.0, help="(Deprecated) No-op; verification is agent-driven")
    ap.add_argument("--log-dir", default="logs", help="Log output directory")

    args = ap.parse_args(argv)

    try:
        workspace = load_workspace(args.workspace)
        if args.agent:
            from .agent import AgentConfig, VisualAutomationAgent

            agent = VisualAutomationAgent(
                workspace=workspace,
                config=AgentConfig(
                    model=args.model,
                    max_steps=args.max_agent_steps,
                    action_delay_s=args.action_delay,
                    log_dir=args.log_dir,
                    abort_hotkey=not args.no_abort_hotkey,
                ),
                dry_run=args.dry_run,
            )
            if args.chat:
                if args.chat_plain:
                    agent.chat(first_message=args.command or None)
                else:
                    from .chat_tui import run_chat_tui

                    return run_chat_tui(
                        agent=agent,
                        first_message=args.command or None,
                    )
            else:
                if not args.command:
                    raise ValueError("Missing --command (or use --chat).")
                agent.run(user_command=args.command)
        else:
            if not args.command:
                raise ValueError("Missing --command.")
            steps = parse_command(args.command)
            _print_plan(steps)
            run_plan(
                workspace=workspace,
                steps=steps,
                dry_run=args.dry_run,
                no_ocr=args.no_ocr,
                require_ocr=args.require_ocr,
                abort_hotkey=not args.no_abort_hotkey,
                action_delay_s=args.action_delay,
                max_attempts=args.max_attempts,
                tolerance_mv=args.tolerance_mv,
                log_dir=args.log_dir,
            )
        return 0
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
