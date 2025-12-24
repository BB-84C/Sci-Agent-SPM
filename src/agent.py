from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from .actions import ActionConfig, Actor
from .abort import start_abort_hotkey
from .capture import ScreenCapturer
from .llm_client import LlmConfig, OpenAiMultimodalClient
from .logger import RunLogger
from .tools import BASE_TOOL_NAMES
from .tools.click_anchor import handle as tool_click_anchor
from .tools.fail import handle as tool_fail
from .tools.finish import handle as tool_finish
from .tools.launch_calibrator import handle as tool_launch_calibrator
from .tools.observe import handle as tool_observe
from .tools.wait_until import handle as tool_wait_until
from .tools.set_field import handle as tool_set_field
from .workspace import Workspace, load_workspace


SYSTEM_PROMPT = """You are an automation agent controlling a Windows desktop app via fixed click anchors and ROI screenshots.
You MUST NOT claim to click UI elements by name; you can only use provided anchors and ROIs.

You operate in ReAct style:
- You see OBSERVATION text + ROI screenshots.
- You choose ONE tool action at a time.
- After each action, you will get a new observation.

Return ONLY JSON with keys:
  - "action": one of ["observe","wait_until","click_anchor","set_field","launch_calibrator","finish","fail"]
  - "action_input": object with parameters for the action
  - "say": short, demo-friendly narration (1-2 sentences, no internal reasoning)
Optional keys (keep short):
  - "plan": list of 1-line steps
  - "observation": short summary of what you see
  - "rationale": short reason for the chosen action (1 sentence)

Guidelines:
- If the user asks to recalibrate ROIs/anchors, choose launch_calibrator.
- If units differ (V vs mV), decide an appropriate numeric entry so the resulting value equals the user's target.
- Prefer observing relevant ROIs after taking actions.
- Avoid repeating actions without checking results. If the ROI looks correct, choose finish.

IMPORTANT:
- Do not rely on OCR. Use the ROI images directly to judge whether values changed as intended.
- For set_field (generic), provide:
  - {"anchor": "<anchor_name>", "typed_text": "<text>", "submit": "enter"|"tab"|null, "rois": ["roi1","roi2"] }
  - You should choose anchor/roi names from the workspace lists in OBSERVATION.
- If a requested action requires an anchor that does not exist, choose launch_calibrator (not fail).
 - For wait_until, provide:
  - {"roi": "<roi_name>", "seconds": <int>, "max_rounds": <int optional>, "max_total_seconds": <int optional>, "reason": "<string optional>"}
"""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-5.2"
    max_steps: int = 10
    action_delay_s: float = 0.25
    log_dir: str = "logs"
    abort_hotkey: bool = True
    memory_turns: int = 6


class VisualAutomationAgent:
    def __init__(
        self,
        *,
        workspace: Workspace,
        config: AgentConfig,
        dry_run: bool,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        logger: Optional[RunLogger] = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.dry_run = dry_run

        self.logger = logger or RunLogger(root_dir=config.log_dir)
        self.capturer = ScreenCapturer()
        self._event_sink = event_sink

        abort = start_abort_hotkey() if config.abort_hotkey else None
        self._abort = abort
        self.actor = Actor(
            dry_run=dry_run,
            config=ActionConfig(delay_s=config.action_delay_s),
            abort_event=abort.event if abort else None,
        )

        self.llm = OpenAiMultimodalClient(LlmConfig(model=config.model))

        self._last_action_log: str = "(none yet)"
        self._memory: list[str] = []
        self._turn: int = 0
        self._consecutive_observes: int = 0
        self._last_action_signature: Optional[str] = None
        self._observed_since_last_action: bool = True
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._tokens_total: int = 0

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, **payload})
        except Exception:
            pass

    def export_session(self) -> Mapping[str, Any]:
        return {
            "memory": list(self._memory),
            "last_action_log": self._last_action_log,
            "tokens": {
                "input_tokens": self._tokens_in,
                "output_tokens": self._tokens_out,
                "total_tokens": self._tokens_total,
            },
        }

    def import_session(self, state: Mapping[str, Any]) -> None:
        mem = state.get("memory", None)
        if isinstance(mem, list) and all(isinstance(x, str) for x in mem):
            self._memory = list(mem)
        self._last_action_log = str(state.get("last_action_log", self._last_action_log))
        toks = state.get("tokens", None)
        if isinstance(toks, dict):
            try:
                self._tokens_in = int(toks.get("input_tokens", self._tokens_in))
                self._tokens_out = int(toks.get("output_tokens", self._tokens_out))
                self._tokens_total = int(toks.get("total_tokens", self._tokens_total))
            except Exception:
                pass
        self._emit(
            "tokens",
            input_tokens=self._tokens_in,
            output_tokens=self._tokens_out,
            total_tokens=self._tokens_total,
        )

    def _accumulate_last_usage(self) -> None:
        usage = getattr(self.llm, "last_usage", None)
        if not isinstance(usage, dict):
            return
        try:
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            total = int(usage.get("total_tokens", in_tok + out_tok))
        except Exception:
            return
        self._tokens_in += max(0, in_tok)
        self._tokens_out += max(0, out_tok)
        self._tokens_total += max(0, total if total > 0 else in_tok + out_tok)
        self._emit(
            "tokens",
            input_tokens=self._tokens_in,
            output_tokens=self._tokens_out,
            total_tokens=self._tokens_total,
        )

    def _tool_names(self) -> list[str]:
        return list(BASE_TOOL_NAMES)

    def _workspace_text(self) -> str:
        rois = "\n".join(f"- {r.name}: {r.description}" for r in self.workspace.rois) or "(none)"
        anchors = "\n".join(f"- {a.name}: {a.description}" for a in self.workspace.anchors) or "(none)"
        return f"ROIs:\n{rois}\n\nAnchors:\n{anchors}"

    def _memory_text(self) -> str:
        if not self._memory:
            return "(empty)"
        keep = self._memory[-max(1, self.config.memory_turns) :]
        return "\n".join(keep)

    def _remember(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        self._memory.append(line)
        if len(self._memory) > max(1, self.config.memory_turns) * 3:
            self._memory = self._memory[-max(1, self.config.memory_turns) * 3 :]

    def _observe_images(self, roi_names: Iterable[str]) -> list[tuple[str, str, Any]]:
        out: list[tuple[str, str, Any]] = []
        for name in roi_names:
            roi = self.workspace.roi(name)
            out.append((name, roi.description, self.capturer.capture_roi(roi)))
        return out

    def _default_observation(self) -> tuple[str, list[tuple[str, str, Any]]]:
        # If the workspace has tool hints, prefer those ROIs; otherwise capture all ROIs.
        rois = [r.name for r in self.workspace.rois]
        obs_text = (
            f"Last action log: {self._last_action_log}\n"
            f"Available ROIs: {', '.join(rois)}\n"
        )
        obs_text = f"{obs_text}\n\n{self._workspace_text()}"
        images = self._observe_images(rois)
        return obs_text, images

    def _log_observation(
        self, *, step_name: str, images: list[tuple[str, str, Any]], meta: Mapping[str, Any]
    ) -> None:
        step = self.logger.start_step(step_name)
        for roi_name, _desc, img in images:
            step.save_image(f"after_{roi_name}.png", img)
        step.write_meta(dict(meta))

    def run(self, *, user_command: str) -> list[Mapping[str, Any]]:
        self._turn += 1
        self._remember(f"User: {user_command}")
        self._emit("user", text=user_command)
        # Reload workspace on each run so chat sessions track user edits (add/delete ROIs/anchors).
        try:
            self.workspace = load_workspace(self.workspace.source_path)
        except Exception:
            pass
        self.logger.narrate(f"Workspace: {self.workspace.source_path}")
        self.logger.narrate(f"Agent mode: True (model={self.config.model})")
        self.logger.narrate(f"Dry-run: {self.dry_run}")
        self.logger.narrate("Failsafe: move mouse to top-left to abort (pyautogui.FAILSAFE)")
        if self.config.abort_hotkey:
            self.logger.narrate("Abort hotkey: ESC")

        results: list[Mapping[str, Any]] = []
        try:
            for i in range(1, self.config.max_steps + 1):
                obs_text, obs_images = self._default_observation()
                self._emit(
                    "observation",
                    step=i,
                    text=obs_text,
                    rois=[{"name": n, "description": d} for (n, d, _img) in obs_images],
                )

                model_out = self.llm.react_step(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"USER COMMAND: {user_command}",
                    memory_text=self._memory_text(),
                    observation_text=obs_text,
                    observation_images=obs_images,
                    tool_names=self._tool_names(),
                )
                self._accumulate_last_usage()

                action = str(model_out.get("action", "")).strip()
                action_input = model_out.get("action_input", {}) if isinstance(model_out.get("action_input", {}), dict) else {}
                say = str(model_out.get("say", "")).strip() or action
                plan = model_out.get("plan", None)
                observation_summary = str(model_out.get("observation", "")).strip() if "observation" in model_out else ""
                rationale = str(model_out.get("rationale", "")).strip() if "rationale" in model_out else ""

                signature = None
                if action in {"wait_until", "click_anchor", "set_field", "launch_calibrator"}:
                    try:
                        items = sorted((str(k), str(v)) for k, v in action_input.items())
                        signature = f"{action}:{items}"
                    except Exception:
                        signature = f"{action}:?"

                if signature is not None and signature == self._last_action_signature:
                    # Force an observation instead of blindly repeating the exact same action.
                    if self._observed_since_last_action:
                        action = "finish"
                        action_input = {}
                        say = "UI appears stable after the last action; stopping to avoid repeating the same step."
                    else:
                        action = "observe"
                        action_input = {"rois": [r.name for r in self.workspace.rois]}
                        say = "Re-checking ROIs after the last action to confirm the current state."

                self.logger.narrate(f"[Agent] Step {i}/{self.config.max_steps}: {say}")
                self._remember(f"Agent: {say}")
                self._emit(
                    "decision",
                    step=i,
                    say=say,
                    action=action,
                    action_input=action_input,
                    plan=plan if isinstance(plan, list) else None,
                    observation=observation_summary,
                    rationale=rationale,
                )

                if action == "observe":
                    flow = tool_observe(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    if flow == "break":
                        break
                    continue

                if action == "wait_until":
                    tool_wait_until(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    continue

                if action == "click_anchor":
                    tool_click_anchor(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    continue

                if action == "set_field":
                    tool_set_field(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    continue

                if action == "launch_calibrator":
                    tool_launch_calibrator(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    break

                if action == "finish":
                    tool_finish(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    break

                if action == "fail":
                    flow = tool_fail(
                        self,
                        step_index=i,
                        action_input=action_input,
                        say=say,
                        signature=signature,
                        results=results,
                    )
                    if flow == "break":
                        break

                raise ValueError(f"Invalid agent action: {action!r}")
        finally:
            if self._abort is not None:
                self._abort.stop()

        self.logger.narrate(f"Logs: {self.logger.run_root}")
        return results

    def chat(self, *, first_message: str | None = None) -> None:
        self.logger.narrate("Chat mode: type a command, or 'exit' to quit.")

        def run_one(text: str) -> None:
            try:
                self.run(user_command=text)
            except KeyboardInterrupt:
                self.logger.narrate("Aborted.")
            except Exception as e:
                # Stay in the chat session even if one command fails.
                self.logger.narrate(f"[Agent/Error] {e}")

        if first_message and first_message.strip():
            run_one(first_message.strip())
        while True:
            try:
                msg = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg:
                continue
            if msg.lower() in {"exit", "quit"}:
                break
            run_one(msg)
