from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from .actions import ActionConfig, Actor
from .abort import start_abort_hotkey
from .capture import ScreenCapturer
from .llm_client import LlmConfig, OpenAiMultimodalClient
from .logger import RunLogger
from .skills.set_bias import SetBiasParams
from .skills.start_scan import StartScanParams
from .workspace import Workspace, load_workspace


SYSTEM_PROMPT = """You are an automation agent controlling a Windows desktop app via fixed click anchors and ROI screenshots.
You MUST NOT claim to click UI elements by name; you can only use provided anchors and ROIs.

You operate in ReAct style:
- You see OBSERVATION text + ROI screenshots.
- You choose ONE tool action at a time.
- After each action, you will get a new observation.

Return ONLY JSON with keys:
  - "action": one of ["observe","click_anchor","set_field","set_bias","start_scan","pause_scan","launch_calibrator","finish","fail"]
  - "action_input": object with parameters for the action
  - "say": short, demo-friendly narration (1-2 sentences, no internal reasoning)
Optional keys (keep short):
  - "plan": list of 1-line steps
  - "observation": short summary of what you see
  - "rationale": short reason for the chosen action (1 sentence)

Guidelines:
- If the user asks to recalibrate ROIs/anchors, choose launch_calibrator.
- If units differ (V vs mV), decide an appropriate numeric entry so the resulting bias equals the user's target.
- Prefer observing relevant ROIs after taking actions.
- Avoid repeating actions without checking results. If the ROI looks correct, choose finish.

IMPORTANT:
- Do not rely on OCR. Use the ROI images directly to judge whether values changed as intended.
- For set_bias, provide either:
  - {"target_value": <number>, "target_unit": "mV"|"V"} and optionally "typed_text", OR
  - {"typed_text": "<number>"} (will be interpreted in the configured input unit).
- For set_field (generic), provide:
  - {"anchor": "<anchor_name>", "typed_text": "<text>", "submit": "enter"|"tab"|null, "rois": ["roi1","roi2"] }
  - You should choose anchor/roi names from the workspace lists in OBSERVATION.
- If a requested action requires an anchor that does not exist, choose launch_calibrator (not fail).
- For pause_scan, provide:
  - {"roi": "scan_time_count_down"} (optional; defaults to that name)
  - The agent will read the remaining scan time from this ROI image, wait that many seconds + 5, then continue.
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
        self._scan_wait_pending: bool = False
        self._scan_wait_used: bool = False
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
            "scan_wait_pending": self._scan_wait_pending,
            "scan_wait_used": self._scan_wait_used,
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
        self._scan_wait_pending = bool(state.get("scan_wait_pending", False))
        self._scan_wait_used = bool(state.get("scan_wait_used", False))
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
        tools = ["observe", "click_anchor", "set_field", "launch_calibrator", "finish", "fail"]
        anchor_names = {a.name for a in self.workspace.anchors}
        if "bias_input" in anchor_names:
            tools.insert(1, "set_bias")
        if (
            "scan_start_button" in anchor_names
            or "scan_start_from_top_button" in anchor_names
            or "scan_start_from_bottom_button" in anchor_names
        ):
            tools.insert(1, "start_scan")
        roi_names = {r.name for r in self.workspace.rois}
        if "scan_time_count_down" in roi_names:
            tools.insert(1, "pause_scan")
        return tools

    def _workspace_text(self) -> str:
        rois = "\n".join(f"- {r.name}: {r.description}" for r in self.workspace.rois) or "(none)"
        anchors = "\n".join(f"- {a.name}: {a.description}" for a in self.workspace.anchors) or "(none)"
        return f"ROIs:\n{rois}\n\nAnchors:\n{anchors}"

    def _memory_text(self) -> str:
        if not self._memory:
            return "(empty)"
        keep = self._memory[-max(1, self.config.memory_turns) :]
        return "\n".join(keep)

    def _infer_bias_input_unit(self) -> str:
        cfg: str = ""
        tool_cfg = (self.workspace.tools or {}).get("SetBias", {})
        if isinstance(tool_cfg, dict):
            cfg = str(tool_cfg.get("input_unit", "")).strip().lower()
        if cfg in {"v", "volt", "volts"}:
            return "V"
        if cfg in {"mv", "millivolt", "millivolts"}:
            return "mV"
        # Default to Volts for Nanonis-style fields unless configured otherwise.
        return "V"

    def _normalize_set_bias_action(self, action_input: Mapping[str, Any]) -> tuple[float, str, Optional[str]]:
        """
        Normalizes model-provided set_bias inputs to prevent unit mistakes.

        Common failure mode: model returns value=700 with unit="V" while typing "0.7" (meaning 700 mV).
        In that case, we trust typed_text (in the configured input unit) for safety/logging.
        """

        typed_text_raw = action_input.get("typed_text", None)
        typed_text = str(typed_text_raw).strip() if typed_text_raw is not None else None
        typed_float: Optional[float] = None
        if typed_text:
            try:
                typed_float = float(typed_text)
            except Exception:
                typed_float = None

        raw_target = action_input.get("value", action_input.get("target_value", None))
        target_value: Optional[float] = None
        if raw_target is not None:
            try:
                target_value = float(raw_target)
            except Exception:
                target_value = None

        unit_raw = action_input.get("unit", action_input.get("target_unit", None))
        unit: Optional[str] = str(unit_raw).strip() if unit_raw is not None else None
        if unit is not None:
            unit = "V" if unit.lower() in {"v", "volt", "volts"} else "mV"

        input_unit = self._infer_bias_input_unit()  # "V" or "mV"

        # If unit is missing, infer from magnitude (bias setpoints are usually <= a few V).
        if unit is None and target_value is not None:
            unit = "V" if abs(target_value) <= 10 else "mV"

        # If still missing, fall back to configured input unit.
        if unit is None:
            unit = input_unit

        # If target_value is missing, derive it from typed_text in the configured input unit.
        if target_value is None and typed_float is not None:
            return typed_float, input_unit, typed_text

        # If we have a target_value but it's clearly inconsistent with typed_text and unit, trust typed_text.
        # Example: value=700, unit="V", typed_text="0.7", input_unit="V" -> interpret as 0.7 V.
        if (
            target_value is not None
            and typed_float is not None
            and unit == input_unit
            and unit == "V"
            and abs(target_value) > 50
            and abs(typed_float) <= 10
        ):
            return typed_float, "V", typed_text

        return float(target_value or 0.0), unit, typed_text

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
            f"SetBias input_unit config: {self._infer_bias_input_unit()}"
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
                if action in {"set_bias", "start_scan", "pause_scan", "launch_calibrator"}:
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

                # If a scan was just started, avoid observe-spam: prefer pause_scan once if countdown ROI exists.
                if (
                    action == "observe"
                    and self._scan_wait_pending
                    and not self._scan_wait_used
                    and any(r.name == "scan_time_count_down" for r in self.workspace.rois)
                ):
                    action = "pause_scan"
                    action_input = {"roi": "scan_time_count_down"}
                    say = "Waiting for the scan countdown to finish before checking status again."

                if action == "observe":
                    self._consecutive_observes += 1
                    roi_names = action_input.get("rois", None)
                    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
                        roi_names = [r.name for r in self.workspace.rois]
                    images = self._observe_images(roi_names)
                    self._last_action_log = f"observe(rois={roi_names})"
                    self._observed_since_last_action = True
                    self._log_observation(step_name=f"agent_observe_{i}", images=images, meta={"action": action, "action_input": action_input, "say": say})
                    results.append({"action": action, "action_input": action_input, "say": say})
                    self._emit("result", step=i, action=action, result={"rois": roi_names, "log_root": str(self.logger.run_root)})
                    if self._consecutive_observes >= 3:
                        self.logger.narrate("[Agent] Observed repeatedly; stopping to avoid loops.")
                        results.append({"action": "finish", "action_input": {}, "say": "Done observing."})
                        self._emit("finish", step=i, say="Done observing.")
                        break
                    continue

                if action == "pause_scan":
                    self._consecutive_observes = 0
                    roi_name = str(action_input.get("roi", "scan_time_count_down")).strip() or "scan_time_count_down"
                    roi = self.workspace.roi(roi_name)

                    step = self.logger.start_step("pause_scan")
                    self.logger.narrate("[PauseScan] Reading scan countdown and waiting…")

                    img = self.capturer.capture_roi(roi)
                    step.save_image(f"countdown_{roi.name}.png", img)

                    parsed = self.llm.extract_countdown_seconds(
                        roi_name=roi.name,
                        roi_description=roi.description,
                        image=img,
                    )
                    self._accumulate_last_usage()
                    seconds_remaining = parsed.get("seconds_remaining", None)
                    wait_s: Optional[float] = None
                    if isinstance(seconds_remaining, (int, float)):
                        wait_s = max(0.0, float(seconds_remaining) + 5.0)

                    meta: dict[str, Any] = {
                        "skill": "PauseScan",
                        "roi": roi.name,
                        "parsed": parsed,
                        "wait_seconds": wait_s,
                        "added_seconds": 5.0,
                        "dry_run": self.dry_run,
                    }
                    step.write_meta(meta)

                    self._scan_wait_used = True
                    self._scan_wait_pending = False

                    self._last_action_log = "pause_scan()"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    results.append({"action": action, "action_input": action_input, "result": meta, "say": say})
                    self._emit("result", step=i, action=action, result=meta | {"log_root": str(self.logger.run_root)})

                    if wait_s is None:
                        self.logger.narrate(
                            "[PauseScan] Could not read countdown; will only re-check a few times to avoid loops."
                        )
                        continue

                    self.logger.narrate(f"[PauseScan] Waiting {wait_s:.1f} seconds (countdown + 5s)…")
                    if not self.dry_run:
                        time.sleep(min(wait_s, 6 * 60 * 60))
                    continue

                if action == "click_anchor":
                    self._consecutive_observes = 0
                    name = str(action_input.get("anchor", "")).strip()
                    if not name:
                        raise ValueError("click_anchor requires action_input.anchor")
                    anchor = self.workspace.anchor(name)
                    step = self.logger.start_step(f"click_{name}")
                    self.logger.narrate(f"[Click] Clicking anchor {name!r}.")
                    actor = self.actor
                    actor.click(anchor)
                    # Save after images for requested rois (or all).
                    roi_names = action_input.get("rois", None)
                    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
                        roi_names = [r.name for r in self.workspace.rois]
                    images = self._observe_images(roi_names)
                    for roi_name, _desc, img in images:
                        step.save_image(f"after_{roi_name}.png", img)
                    step.write_meta({"action": "click_anchor", "anchor": name, "rois": roi_names, "say": say})
                    self._last_action_log = f"click_anchor(anchor={name})"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    results.append({"action": "click_anchor", "action_input": action_input, "say": say})
                    self._emit("result", step=i, action=action, result={"anchor": name, "rois": roi_names, "log_root": str(self.logger.run_root)})
                    continue

                if action == "set_field":
                    self._consecutive_observes = 0
                    name = str(action_input.get("anchor", "")).strip()
                    typed_text = str(action_input.get("typed_text", "")).strip()
                    submit = action_input.get("submit", "enter")
                    submit_key = None if submit is None else str(submit).strip().lower()
                    roi_names = action_input.get("rois", None)
                    if not isinstance(roi_names, list) or not all(isinstance(x, str) for x in roi_names):
                        roi_names = [r.name for r in self.workspace.rois]
                    if not name or not typed_text:
                        raise ValueError("set_field requires action_input.anchor and action_input.typed_text")

                    anchor = self.workspace.anchor(name)
                    step = self.logger.start_step(f"set_field_{name}")
                    self.logger.narrate(f"[SetField] Setting {name!r} by typing {typed_text!r}.")

                    before_imgs = self._observe_images(roi_names)
                    for roi_name, _desc, img in before_imgs:
                        step.save_image(f"before_{roi_name}.png", img)

                    actor = self.actor
                    actor.double_click(anchor)
                    actor.hotkey("ctrl", "a")
                    actor.press("backspace")
                    actor.press("backspace")
                    actor.type_text(typed_text)
                    if submit_key in {"enter", "return"}:
                        actor.press("enter")
                    elif submit_key in {"tab"}:
                        actor.press("tab")

                    after_imgs = self._observe_images(roi_names)
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
                    self._last_action_log = f"set_field(anchor={name}, typed_text={typed_text!r})"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    results.append({"action": "set_field", "action_input": action_input, "say": say})
                    self._emit("result", step=i, action=action, result={"anchor": name, "typed_text": typed_text, "rois": roi_names, "log_root": str(self.logger.run_root)})
                    continue

                if action == "set_bias":
                    self._consecutive_observes = 0
                    target_value, unit, typed_text = self._normalize_set_bias_action(action_input)
                    params = SetBiasParams(target_value=float(target_value), target_unit=unit, typed_text=typed_text)
                    from .skills.set_bias import run as run_set_bias

                    meta = run_set_bias(
                        workspace=self.workspace,
                        capturer=self.capturer,
                        actor=self.actor,
                        logger=self.logger,
                        params=params,
                    )
                    self._last_action_log = f"set_bias(value={target_value}, unit={unit})"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    results.append({"action": action, "action_input": action_input, "result": meta, "say": say})
                    self._emit("result", step=i, action=action, result=meta | {"log_root": str(self.logger.run_root)})
                    continue

                if action == "start_scan":
                    self._consecutive_observes = 0
                    direction = str(action_input.get("direction", "default")).lower()
                    if direction not in {"top", "bottom", "default"}:
                        direction = "default"
                    params = StartScanParams(direction=direction)  # type: ignore[arg-type]
                    from .skills.start_scan import run as run_start_scan

                    meta = run_start_scan(
                        workspace=self.workspace,
                        capturer=self.capturer,
                        actor=self.actor,
                        logger=self.logger,
                        params=params,
                    )
                    self._last_action_log = f"start_scan(direction={direction})"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    self._scan_wait_pending = True
                    self._scan_wait_used = False
                    results.append({"action": action, "action_input": action_input, "result": meta, "say": say})
                    self._emit("result", step=i, action=action, result=meta | {"log_root": str(self.logger.run_root)})
                    continue

                if action == "launch_calibrator":
                    self._consecutive_observes = 0
                    # Launch GUI calibrator and stop (user must adjust + save).
                    ws_path = str(self.workspace.source_path)
                    cmd = [sys.executable, "-m", "src.calibrate_gui", "--workspace", ws_path]
                    try:
                        subprocess.Popen(cmd, close_fds=True)
                    except Exception:
                        pass
                    self._last_action_log = "launch_calibrator()"
                    self._last_action_signature = signature
                    self._observed_since_last_action = False
                    self.logger.narrate(f"[Agent] Launched calibrator for {ws_path}. Save and rerun the command.")
                    results.append({"action": action, "action_input": action_input, "say": say})
                    self._emit("result", step=i, action=action, result={"workspace": ws_path})
                    break

                if action == "finish":
                    self._consecutive_observes = 0
                    self._last_action_signature = None
                    self._observed_since_last_action = True
                    results.append({"action": action, "action_input": action_input, "say": say})
                    self._emit("finish", step=i, say=say)
                    break

                if action == "fail":
                    self._consecutive_observes = 0
                    self._last_action_signature = None
                    self._observed_since_last_action = True
                    msg = str(action_input.get("message", "Agent failed.")).strip()
                    # Prefer offering recalibration over hard failing when missing UI mappings.
                    msg_l = msg.lower()
                    if any(t in msg_l for t in ["anchor", "roi", "calibrat", "mapping", "not able", "can't set"]):
                        ws_path = str(self.workspace.source_path)
                        self.logger.narrate(f"[Agent] {msg} Launching calibrator to update workspace mappings…")
                        cmd = [sys.executable, "-m", "src.calibrate_gui", "--workspace", ws_path]
                        try:
                            subprocess.Popen(cmd, close_fds=True)
                        except Exception:
                            pass
                        results.append({"action": "launch_calibrator", "action_input": {"workspace": ws_path}, "say": msg})
                        self._emit("result", step=i, action="launch_calibrator", result={"workspace": ws_path, "reason": msg})
                        break
                    raise RuntimeError(msg)

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
