from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from .actions import ActionConfig, Actor
from .abort import start_abort_hotkey
from .capture import ScreenCapturer
from .llm_client import LlmConfig, OpenAiMultimodalClient
from .logger import RunLogger
from .mcp_server import McpToolContext, create_mcp_server
from .workspace import Workspace, load_workspace


SYSTEM_PROMPT = """You are an automation agent controlling a Windows desktop app via ROI screenshots and fixed click anchors.
You MUST NOT claim to click UI elements by name; you can only use provided anchors and ROIs.

You operate in ReAct style:
- You see OBSERVATION text + ROI screenshots.
- You call exactly ONE available tool per step.
- After each tool call, you will receive a new observation.

Keep any narration short (1–2 sentences). Do not reveal internal reasoning.

Guidelines:
- If the user asks to recalibrate ROIs/anchors, call the calibrator tool.
- Prefer observing relevant ROIs after taking actions.
- Avoid repeating actions without checking results. If the UI looks correct, finish.

IMPORTANT:
- Use the ROI images directly to judge whether values changed as intended.
- Choose anchor/ROI names from the workspace lists in OBSERVATION.
- If a requested action requires an anchor or ROI that does not exist, call the calibrator tool (do not fail).
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
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        logger: Optional[RunLogger] = None,
    ) -> None:
        self.workspace = workspace
        self.config = config

        self.logger = logger or RunLogger(root_dir=config.log_dir)
        self.capturer = ScreenCapturer()
        self._event_sink = event_sink

        abort = start_abort_hotkey() if config.abort_hotkey else None
        self._abort = abort
        self.actor = Actor(
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

        self._mcp = create_mcp_server(agent=self)
        self._openai_tools_cache: Optional[list[dict[str, Any]]] = None
        self._mcp_tool_ctx: Optional[McpToolContext] = None

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, **payload})
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("Model name cannot be empty.")
        self.config = replace(self.config, model=model)
        try:
            self.llm._config = replace(self.llm._config, model=model)  # type: ignore[attr-defined]
        except Exception:
            self.llm = OpenAiMultimodalClient(LlmConfig(model=model))

    def set_max_steps(self, max_steps: int) -> None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        self.config = replace(self.config, max_steps=max_steps)

    def set_action_delay_s(self, delay_s: float) -> None:
        delay_s = float(delay_s)
        if delay_s < 0:
            raise ValueError("action_delay_s must be >= 0.")
        self.config = replace(self.config, action_delay_s=delay_s)
        self.actor.set_delay_s(delay_s)

    def set_abort_hotkey(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.config = replace(self.config, abort_hotkey=enabled)

        if enabled:
            if self._abort is None:
                self._abort = start_abort_hotkey()
                self.actor.set_abort_event(self._abort.event)
            return

        if self._abort is not None:
            try:
                self._abort.stop()
            finally:
                self._abort = None
        self.actor.set_abort_event(None)

    def set_log_dir(self, log_dir: str) -> None:
        log_dir = log_dir.strip()
        if not log_dir:
            raise ValueError("log_dir cannot be empty.")
        self.config = replace(self.config, log_dir=log_dir)
        self.logger = RunLogger(root_dir=log_dir)

    def set_workspace(self, workspace_path: str) -> None:
        ws = load_workspace(workspace_path)
        self.workspace = ws

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

    def _openai_tools(self) -> list[dict[str, Any]]:
        if self._openai_tools_cache is not None:
            return list(self._openai_tools_cache)

        async def _list() -> list[dict[str, Any]]:
            async with create_connected_server_and_client_session(self._mcp) as mcp_session:
                mcp_tools = await mcp_session.list_tools()
            openai_tools: list[dict[str, Any]] = []
            for t in mcp_tools.tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                        "strict": False,
                    }
                )
            return openai_tools

        self._openai_tools_cache = anyio.run(_list)
        return list(self._openai_tools_cache)

    def _call_mcp_tool(
        self,
        *,
        tool_name: str,
        tool_args: Mapping[str, Any],
        step_index: int,
        say: str,
        signature: Optional[str],
        results: list[dict[str, Any]],
    ) -> Literal["continue", "break"]:
        self._mcp_tool_ctx = McpToolContext(step_index=step_index, say=say, signature=signature, results=results)

        async def _call() -> str:
            async with create_connected_server_and_client_session(self._mcp) as mcp_session:
                res = await mcp_session.call_tool(tool_name, dict(tool_args))
            for item in res.content:
                if getattr(item, "type", None) == "text":
                    return str(getattr(item, "text", "")).strip()
            return "continue"

        try:
            flow = anyio.run(_call).strip().lower()
        finally:
            self._mcp_tool_ctx = None

        return "break" if flow == "break" else "continue"

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
        self.logger.narrate("Failsafe: move mouse to top-left to abort (pyautogui.FAILSAFE)")
        if self.config.abort_hotkey:
            self.logger.narrate("Abort hotkey: ESC")

        results: list[dict[str, Any]] = []
        openai_tools = self._openai_tools()
        known_tools = {t.get("name") for t in openai_tools}

        plan: list[str] = []
        try:
            plan_out = self.llm.plan_step(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_command,
                memory_text=self._memory_text(),
                workspace_text=self._workspace_text(),
            )
            self._accumulate_last_usage()
            raw_plan = plan_out.get("plan", None)
            if isinstance(raw_plan, list) and all(isinstance(x, str) for x in raw_plan):
                plan = [x.strip() for x in raw_plan if x.strip()]
        except Exception:
            plan = []

        if plan:
            self._remember("Plan:\n" + "\n".join(f"- {x}" for x in plan))
            self._emit("plan", plan=plan)
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
                    plan_text="\n".join(f"- {x}" for x in plan) if plan else "(none)",
                    observation_text=obs_text,
                    observation_images=obs_images,
                    tools=openai_tools,
                )
                self._accumulate_last_usage()

                say = str(model_out.get("text", "")).strip()
                tool_calls = model_out.get("tool_calls", [])
                if not isinstance(tool_calls, list) or not tool_calls:
                    tool_calls = [{"name": "finish", "arguments": {}}]
                tool_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {"name": "finish", "arguments": {}}

                action = str(tool_call.get("name", "")).strip()
                action_input = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments", {}), dict) else {}
                if not say:
                    say = action or "(tool)"

                if action not in known_tools:
                    raise ValueError(f"Unknown tool requested by model: {action!r}")

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
                    plan=None,
                    observation="",
                    rationale="",
                )

                flow = self._call_mcp_tool(
                    tool_name=action,
                    tool_args=action_input,
                    step_index=i,
                    say=say,
                    signature=signature,
                    results=results,
                )
                if flow == "break":
                    break
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
