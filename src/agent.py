from __future__ import annotations

import hashlib
import json
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
- If the user asks to recalibrate ROIs/anchors, call the calibration tool.
- Prefer observing relevant ROIs after taking actions.
- If an anchor declares linked_ROIs, use those ROIs for post-action verification (and for deciding what to wait on).
- If the user asks you to report a numeric readout (e.g., bias/current/countdown), do not finish until you have either reported the value(s) or explicitly said the ROI is unreadable and asked to recalibrate/expand it.
- Avoid repeating actions without checking results. If the UI looks correct, stop using a terminal tool.

IMPORTANT:
- Use the ROI images directly to judge whether values changed as intended.
- Choose anchor/ROI names from the workspace lists in OBSERVATION.
- If a requested action requires an anchor or ROI that does not exist, call the calibration tool (do not stop with an error).
"""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    agent_model: str = "gpt-5.2"
    tool_call_model: str = "gpt-5.2"
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

        self.llm_agent = OpenAiMultimodalClient(LlmConfig(model=config.agent_model))
        self.llm_tool = OpenAiMultimodalClient(LlmConfig(model=config.tool_call_model))
        # Back-compat alias used by some tool implementations.
        self.llm = self.llm_agent

        self._last_action_log: str = "(none yet)"
        self._memory: list[str] = []
        self._turn: int = 0
        self._consecutive_observes: int = 0
        self._last_action_signature: Optional[str] = None
        self._observed_since_last_action: bool = True
        self._last_observation_fingerprint: Optional[str] = None
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._tokens_total: int = 0
        self._last_readouts: dict[str, str] = {}
        self._last_unreadable_readouts: list[str] = []

        self._mcp = create_mcp_server(agent=self)
        self._openai_tools_cache: Optional[list[dict[str, Any]]] = None
        self._tool_meta_by_name: dict[str, Mapping[str, Any]] = {}
        self._mcp_tool_ctx: Optional[McpToolContext] = None

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink({"type": event_type, **payload})
        except Exception:
            pass

    def set_model(self, model: str) -> None:
        # Backwards compatibility: treat `/model` as `/agent_model`.
        self.set_agent_model(model)

    def set_agent_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("Agent model name cannot be empty.")
        self.config = replace(self.config, agent_model=model)
        self.llm_agent = OpenAiMultimodalClient(LlmConfig(model=model))
        self.llm = self.llm_agent

    def set_tool_call_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("Tool-call model name cannot be empty.")
        self.config = replace(self.config, tool_call_model=model)
        self.llm_tool = OpenAiMultimodalClient(LlmConfig(model=model))

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

    def _accumulate_last_usage(self, client: Optional[OpenAiMultimodalClient] = None) -> None:
        usage = getattr(client or self.llm, "last_usage", None)
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
                self._tool_meta_by_name[t.name] = dict(t.meta or {})
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

    def _pick_terminal_tool_name(self) -> str:
        if self._openai_tools_cache is None:
            self._openai_tools()
        preferred: Optional[str] = None
        for name, meta in self._tool_meta_by_name.items():
            try:
                md = dict(meta)
                if not bool(md.get("terminal", False)):
                    continue
                if not bool(md.get("error", False)):
                    return str(name)
                if preferred is None:
                    preferred = str(name)
            except Exception:
                continue
        if preferred is not None:
            return preferred
        if self._tool_meta_by_name:
            return next(iter(self._tool_meta_by_name.keys()))
        raise RuntimeError("No tools available.")

    def _call_signature(self, *, tool_name: str, tool_args: Mapping[str, Any]) -> str:
        try:
            normalized = json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            try:
                normalized = str(sorted((str(k), str(v)) for k, v in tool_args.items()))
            except Exception:
                normalized = "<?>"
        return f"{tool_name}:{normalized}"

    def _observation_fingerprint(self, images: Iterable[tuple[str, str, Any]]) -> str:
        """
        Fingerprint the current UI state using a downscaled byte signature of ROI images.
        This avoids treating blank lines in observation text as state changes.
        """
        h = hashlib.sha1()
        for name, _desc, img in images:
            h.update(str(name).encode("utf-8", errors="ignore"))
            try:
                im = img.convert("RGB").resize((24, 24))
                h.update(im.tobytes())
            except Exception:
                h.update(repr(img).encode("utf-8", errors="ignore"))
        return h.hexdigest()

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
        anchor_lines: list[str] = []
        for a in self.workspace.anchors:
            linked = ", ".join(a.linked_rois) if a.linked_rois else ""
            suffix = f" (linked_ROIs: {linked})" if linked else ""
            anchor_lines.append(f"- {a.name}: {a.description}{suffix}")
        anchors = "\n".join(anchor_lines) or "(none)"
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

    def _is_readout_roi(self, name: str) -> bool:
        try:
            roi = self.workspace.roi(name)
        except Exception:
            return False
        n = (roi.name or "").lower()
        d = (roi.description or "").lower()
        tags = [str(t).lower() for t in (roi.tags or ())]
        return ("readout" in n) or ("readout" in d) or ("readout" in tags)

    def _refresh_parsed_readouts_from_images(self, images: list[tuple[str, str, Any]]) -> None:
        readout_items = [(n, d, img) for (n, d, img) in images if self._is_readout_roi(n)]
        if not readout_items:
            return
        try:
            extracted = self.llm_tool.extract_readouts(roi_items=readout_items)
            self._accumulate_last_usage(self.llm_tool)
        except Exception:
            return

        vals = extracted.get("values", {})
        unread = extracted.get("unreadable", [])

        merged = dict(getattr(self, "_last_readouts", {}) or {})
        if isinstance(vals, dict):
            for k, v in vals.items():
                ks = str(k).strip()
                if not ks:
                    continue
                if v is None:
                    continue
                merged[ks] = str(v).strip()

        unread_list: list[str] = []
        if isinstance(unread, list):
            unread_list = [str(x).strip() for x in unread if str(x).strip()]
        # Avoid stale values: if a readout is flagged unreadable, drop any prior value.
        for k in unread_list:
            merged.pop(k, None)

        self._last_readouts = merged
        self._last_unreadable_readouts = unread_list

    def _default_observation(self) -> tuple[str, list[tuple[str, str, Any]]]:
        # If the workspace has tool hints, prefer those ROIs; otherwise capture all ROIs.
        rois = [r.name for r in self.workspace.rois]
        readout_lines: list[str] = []
        for k, v in (self._last_readouts or {}).items():
            if k and v:
                readout_lines.append(f"- {k}: {v}")
        for k in (self._last_unreadable_readouts or []):
            if k:
                readout_lines.append(f"- {k}: (unreadable; consider recalibrating/expanding this ROI)")
        readouts_text = ("\nLast parsed readouts:\n" + "\n".join(readout_lines) + "\n") if readout_lines else ""
        images = self._observe_images(rois)
        # Keep parsed readouts in sync with the current ROI screenshots so narration doesn't repeat stale values.
        self._refresh_parsed_readouts_from_images(images)

        readout_lines = []
        for k, v in (self._last_readouts or {}).items():
            if k and v:
                readout_lines.append(f"- {k}: {v}")
        for k in (self._last_unreadable_readouts or []):
            if k:
                readout_lines.append(f"- {k}: (unreadable; consider recalibrating/expanding this ROI)")
        readouts_text = ("\nLast parsed readouts:\n" + "\n".join(readout_lines) + "\n") if readout_lines else ""

        obs_text = f"Last action log: {self._last_action_log}\nAvailable ROIs: {', '.join(rois)}\n{readouts_text}"
        obs_text = f"{obs_text}\n\n{self._workspace_text()}"
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
        self.logger.narrate(
            f"Agent mode: True (agent_model={self.config.agent_model}, tool_call_model={self.config.tool_call_model})"
        )
        self.logger.narrate("Failsafe: move mouse to top-left to abort (pyautogui.FAILSAFE)")
        if self.config.abort_hotkey:
            self.logger.narrate("Abort hotkey: ESC")

        results: list[dict[str, Any]] = []
        openai_tools = self._openai_tools()
        known_tools = {t.get("name") for t in openai_tools}
        terminal_tool = self._pick_terminal_tool_name()

        def tool_schemas_text() -> str:
            parts: list[str] = []
            for t in openai_tools:
                name = str(t.get("name", "")).strip()
                desc = str(t.get("description", "")).strip()
                params = t.get("parameters", {})
                try:
                    import json as _json

                    schema = _json.dumps(params, ensure_ascii=False)
                except Exception:
                    schema = "{}"
                if not name:
                    continue
                line = f"- {name}: {desc}\n  inputSchema: {schema}"
                parts.append(line)
            return "\n".join(parts) if parts else "(none)"

        plan: list[str] = []
        planned_steps: list[dict[str, Any]] = []
        try:
            plan_out = self.llm_agent.plan_step(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_command,
                memory_text=self._memory_text(),
                workspace_text=self._workspace_text(),
                tools_text=tool_schemas_text(),
            )
            self._accumulate_last_usage(self.llm_agent)
            raw_plan = plan_out.get("plan", None)
            if isinstance(raw_plan, list) and all(isinstance(x, str) for x in raw_plan):
                plan = [x.strip() for x in raw_plan if x.strip()]

            raw_steps = plan_out.get("steps", None)
            if isinstance(raw_steps, list):
                for step in raw_steps:
                    if not isinstance(step, dict):
                        continue
                    tool = step.get("tool", None)
                    args = step.get("args", None)
                    if not isinstance(tool, str) or not tool.strip():
                        continue
                    if not isinstance(args, dict):
                        continue
                    planned_steps.append({"tool": tool.strip(), "args": dict(args), "purpose": str(step.get("purpose", "")).strip()})
        except Exception:
            plan = []
            planned_steps = []

        if plan:
            self._remember("Plan:\n" + "\n".join(f"- {x}" for x in plan))
            self._emit("plan", plan=plan)
        try:
            step_ptr = 0
            for i in range(1, self.config.max_steps + 1):
                obs_text, obs_images = self._default_observation()
                obs_fp = self._observation_fingerprint(obs_images)
                self._emit(
                    "observation",
                    step=i,
                    text=obs_text,
                    rois=[{"name": n, "description": d} for (n, d, _img) in obs_images],
                )

                if planned_steps:
                    if step_ptr >= len(planned_steps):
                        action = terminal_tool
                        action_input: dict[str, Any] = {}
                    else:
                        cur = planned_steps[step_ptr]
                        action = str(cur.get("tool", "")).strip()
                        action_input = dict(cur.get("args", {}) if isinstance(cur.get("args", {}), dict) else {})
                else:
                    model_out = self.llm_tool.react_step(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=f"USER COMMAND: {user_command}",
                        memory_text=self._memory_text(),
                        plan_text="\n".join(f"- {x}" for x in plan) if plan else "(none)",
                        observation_text=obs_text,
                        observation_images=obs_images,
                        tools=openai_tools,
                    )
                    self._accumulate_last_usage(self.llm_tool)

                    tool_calls = model_out.get("tool_calls", [])
                    if not isinstance(tool_calls, list) or not tool_calls:
                        tool_calls = [{"name": terminal_tool, "arguments": {}}]
                    tool_call = (
                        tool_calls[0]
                        if isinstance(tool_calls[0], dict)
                        else {"name": terminal_tool, "arguments": {}}
                    )

                    action = str(tool_call.get("name", "")).strip()
                    action_input = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments", {}), dict) else {}

                if action not in known_tools:
                    raise ValueError(f"Unknown tool requested by model: {action!r}")

                signature = self._call_signature(tool_name=action, tool_args=action_input)
                plan_exhausted = bool(planned_steps) and step_ptr >= len(planned_steps)

                # Generic loop-avoidance: if the model repeats the exact same tool call while the UI
                # (as seen via ROIs) hasn't changed, re-ask once for a different call; otherwise stop.
                if (
                    self._last_action_signature is not None
                    and self._last_observation_fingerprint is not None
                    and signature == self._last_action_signature
                    and obs_fp == self._last_observation_fingerprint
                ):
                    if planned_steps:
                        say = "UI appears unchanged and the same action is repeating; stopping to avoid loops."
                        results.append({"action": terminal_tool, "action_input": {}, "say": say})
                        self._emit("finish", step=i, say=say)
                        break

                    retry_out = self.llm_tool.react_step(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=(
                            f"USER COMMAND: {user_command}\n\n"
                            "Constraint: Do NOT repeat the immediately previous tool call (same tool and same arguments)."
                        ),
                        memory_text=self._memory_text(),
                        plan_text="\n".join(f"- {x}" for x in plan) if plan else "(none)",
                        observation_text=obs_text,
                        observation_images=obs_images,
                        tools=openai_tools,
                    )
                    self._accumulate_last_usage(self.llm_tool)
                    retry_calls = retry_out.get("tool_calls", [])
                    if isinstance(retry_calls, list) and retry_calls and isinstance(retry_calls[0], dict):
                        retry_action = str(retry_calls[0].get("name", "")).strip()
                        retry_args = retry_calls[0].get("arguments", {})
                        if isinstance(retry_args, dict) and retry_action in known_tools:
                            retry_sig = self._call_signature(tool_name=retry_action, tool_args=retry_args)
                            if retry_sig != signature:
                                action = retry_action
                                action_input = dict(retry_args)
                                signature = retry_sig
                            else:
                                say = "UI appears unchanged and the model is repeating the same action; stopping to avoid loops."
                                results.append({"action": terminal_tool, "action_input": {}, "say": say})
                                self._emit("finish", step=i, say=say)
                                break
                    else:
                        say = "UI appears unchanged and the model did not provide a different action; stopping to avoid loops."
                        results.append({"action": terminal_tool, "action_input": {}, "say": say})
                        self._emit("finish", step=i, say=say)
                        break

                # Record state for the next loop detection check.
                self._last_action_signature = signature
                self._last_observation_fingerprint = obs_fp

                narration: Mapping[str, Any] = {}
                say = ""
                plan_block = None
                observation_summary = ""
                rationale = ""
                if plan_exhausted and action == terminal_tool and not action_input:
                    parts: list[str] = []
                    for k, v in (self._last_readouts or {}).items():
                        if k and v:
                            parts.append(f"{k}={v}")
                        if len(parts) >= 4:
                            break
                    if self._last_unreadable_readouts:
                        parts.append(f"unreadable={', '.join(self._last_unreadable_readouts[:3])}")
                    readouts_inline = "; ".join(parts) if parts else ""
                    say = "All planned steps are complete. Finishing now."
                    if readouts_inline:
                        say = f"{say} Final readouts: {readouts_inline}."
                    observation_summary = say
                    rationale = "Planned tool steps are complete, so the run can stop."
                    plan_block = []
                else:
                    try:
                        narration = self.llm_agent.narrate_tool_call(
                            system_prompt=SYSTEM_PROMPT,
                            user_command=user_command,
                            plan_text="\n".join(f"- {x}" for x in plan) if plan else "(none)",
                            memory_text=self._memory_text(),
                            observation_text=obs_text,
                            tool_name=action,
                            tool_args=action_input,
                        )
                        self._accumulate_last_usage(self.llm_agent)
                    except Exception:
                        narration = {}

                    say = str(narration.get("say", "")).strip() or action
                    if isinstance(narration.get("plan", None), list) and all(isinstance(x, str) for x in narration["plan"]):
                        plan_block = [x.strip() for x in narration["plan"] if isinstance(x, str) and x.strip()]
                    observation_summary = str(narration.get("observation", "")).strip()
                    rationale = str(narration.get("rationale", "")).strip()

                self.logger.narrate(f"[Agent] Step {i}/{self.config.max_steps}: {say}")
                self._remember(f"Agent: {say}")
                self._emit(
                    "decision",
                    step=i,
                    say=say,
                    action=action,
                    action_input=action_input,
                    plan=plan_block,
                    observation=observation_summary,
                    rationale=rationale,
                )

                flow = self._call_mcp_tool(
                    tool_name=action,
                    tool_args=action_input,
                    step_index=i,
                    say=say,
                    signature=signature,
                    results=results,
                )
                self._remember(f"Tool: {action} args={action_input}")
                if planned_steps:
                    step_ptr += 1
                if flow == "break":
                    break
            else:
                # Step limit reached without a terminal tool: assess progress and finish gracefully.
                obs_text, _obs_images = self._default_observation()
                executed = results[-8:]
                executed_lines: list[str] = []
                for r in executed:
                    try:
                        executed_lines.append(
                            f"- {str(r.get('action', ''))} args={str(r.get('action_input', {}))} say={str(r.get('say', '')).strip()}"
                        )
                    except Exception:
                        continue
                executed_text = "\n".join(executed_lines) if executed_lines else "(none)"
                assessment_say = (
                    f"Reached the maximum of {self.config.max_steps} steps before finishing. "
                    "I will stop here and report the current progress. "
                    "If you want me to continue, increase /max_agent_steps or ask me to proceed."
                )
                try:
                    plan_text = "\n".join(f"- {x}" for x in plan) if plan else "(none)"
                    assess = self.llm_agent.assess_step_limit(
                        system_prompt=SYSTEM_PROMPT,
                        user_command=user_command,
                        plan_text=plan_text,
                        memory_text=self._memory_text(),
                        observation_text=obs_text,
                        executed_steps_text=executed_text,
                        max_steps=int(self.config.max_steps),
                    )
                    self._accumulate_last_usage(self.llm_agent)
                    assessment_say = str(assess.get("say", "")).strip() or assessment_say
                except Exception:
                    pass
                assessment_say = f"Stopped: reached max agent steps ({self.config.max_steps}). {assessment_say}".strip()

                finish_name = "finish" if "finish" in known_tools else terminal_tool
                finish_sig = self._call_signature(tool_name=finish_name, tool_args={})
                self._call_mcp_tool(
                    tool_name=finish_name,
                    tool_args={},
                    step_index=int(self.config.max_steps),
                    say=assessment_say,
                    signature=finish_sig,
                    results=results,
                )
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
