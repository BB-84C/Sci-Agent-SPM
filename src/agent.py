from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from .actions import ActionConfig, Actor
from .abort import start_abort_hotkey
from .capture import ScreenCapturer
from .llm_client import LlmConfig, OpenAiMultimodalClient, estimate_tokens
from .logger import RunLogger
from .mcp_server import McpToolContext, create_mcp_server
from .workspace import Workspace, load_workspace


SYSTEM_PROMPT = """You are an automation agent controlling a Windows desktop app via ROI screenshots and fixed click anchors.
You MUST NOT claim to click UI elements by name; you can only use provided anchors and ROIs.

You operate in strict ReAct style:
- Action (MCP tool) -> Observe (linked ROIs) -> Think -> Next action
- You call exactly ONE MCP tool per step.
- Observation is provided as ROI screenshots + ROI descriptions.

Keep any narration short (1-2 sentences). Do not reveal hidden chain-of-thought.
"""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    agent_model: str = "gpt-5.2"
    tool_call_model: str = "gpt-5-nano"
    max_steps: int = 50
    action_delay_s: float = 0.25
    log_dir: str = "logs"
    abort_hotkey: bool = True
    run_mode: Literal["agent", "chat", "auto"] = "agent"
    # -1: pass full structured memory; otherwise keep last N entries.
    memory_turns: int = -1
    # Internal observe retry count for unreadable ROIs.
    observe_max_retries: int = 5
    # Auto-compress memory when estimated tokens exceed this threshold.
    memory_compress_threshold_tokens: int = 300_000


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

        abort = start_abort_hotkey(key="none") if config.abort_hotkey else None
        self._abort = abort
        self.actor = Actor(
            config=ActionConfig(delay_s=config.action_delay_s),
            abort_event=abort.event if abort else None,
        )

        self.llm_agent = OpenAiMultimodalClient(LlmConfig(model=config.agent_model))
        self.llm_tool = OpenAiMultimodalClient(LlmConfig(model=config.tool_call_model))
        # Back-compat alias used by some tool implementations.
        self.llm = self.llm_agent

        self._plan_text: str = "(none)"
        self._memory: list[dict[str, Any]] = []
        self._archive_memory: list[dict[str, Any]] = []
        self._run_index: int = 0

        self._last_action_log: str = "(none yet)"
        # Tool back-compat fields (not used by the new orchestrator logic).
        self._consecutive_observes: int = 0
        self._last_action_signature: Optional[str] = None
        self._observed_since_last_action: bool = True
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._tokens_total: int = 0

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

    def _abort_is_set(self) -> bool:
        try:
            return bool(self._abort is not None and self._abort.event.is_set())
        except Exception:
            return False

    def _clear_abort(self) -> None:
        try:
            if self._abort is not None:
                self._abort.event.clear()
        except Exception:
            pass

    def _raise_if_aborted(self) -> None:
        if self._abort_is_set():
            raise KeyboardInterrupt()

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

    def set_run_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if m not in {"agent", "chat", "auto"}:
            raise ValueError("mode must be one of: agent | chat | auto")
        self.config = replace(self.config, run_mode=m)  # type: ignore[arg-type]

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
                self._abort = start_abort_hotkey(key="none")
                self.actor.set_abort_event(self._abort.event)
            return

        if self._abort is not None:
            try:
                self._abort.stop()
            finally:
                self._abort = None
        self.actor.set_abort_event(None)

    def request_abort(self) -> None:
        """
        Request a cooperative abort for the current run (if supported).

        This sets the internal abort event used by the Actor and other long-running loops.
        """
        if self._abort is None:
            if not bool(self.config.abort_hotkey):
                return
            self._abort = start_abort_hotkey(key="none")
            self.actor.set_abort_event(self._abort.event)
        try:
            self._abort.event.set()
        except Exception:
            pass

    def set_log_dir(self, log_dir: str) -> None:
        log_dir = log_dir.strip()
        if not log_dir:
            raise ValueError("log_dir cannot be empty.")
        self.config = replace(self.config, log_dir=log_dir)
        self.logger = RunLogger(root_dir=log_dir)

    def set_workspace(self, workspace_path: str) -> None:
        self.workspace = load_workspace(workspace_path)

    def set_memory_turns(self, memory_turns: int) -> None:
        memory_turns = int(memory_turns)
        if memory_turns < -1:
            raise ValueError("memory_turns must be -1 (full) or >= 0.")
        self.config = replace(self.config, memory_turns=memory_turns)

    def set_memory_compress_threshold_tokens(self, threshold_tokens: int) -> None:
        threshold_tokens = int(threshold_tokens)
        if threshold_tokens < 0:
            raise ValueError("memory_compress_threshold_tokens must be >= 0.")
        self.config = replace(self.config, memory_compress_threshold_tokens=threshold_tokens)

    def export_session(self) -> Mapping[str, Any]:
        return {
            "plan_text": self._plan_text,
            "memory": list(self._memory),
            "archive_memory": list(self._archive_memory),
            "run_index": int(self._run_index),
            "last_action_log": self._last_action_log,
            "tokens": {
                "input_tokens": self._tokens_in,
                "output_tokens": self._tokens_out,
                "total_tokens": self._tokens_total,
            },
        }

    def import_session(self, state: Mapping[str, Any]) -> None:
        self._plan_text = str(state.get("plan_text", self._plan_text))
        mem = state.get("memory", None)
        if isinstance(mem, list) and all(isinstance(x, dict) for x in mem):
            self._memory = [dict(x) for x in mem]
        arch = state.get("archive_memory", None)
        if isinstance(arch, list) and all(isinstance(x, dict) for x in arch):
            self._archive_memory = [dict(x) for x in arch]
        try:
            self._run_index = int(state.get("run_index", self._run_index))
        except Exception:
            pass
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
                name = str(t.name or "").strip()
                self._tool_meta_by_name[name] = dict(t.meta or {})
                # Internal-only / UI-only tools:
                if name in {"observe", "launch_calibrator"}:
                    continue
                if not name:
                    continue
                openai_tools.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                        "strict": False,
                    }
                )
            return openai_tools

        self._openai_tools_cache = anyio.run(_list)
        return list(self._openai_tools_cache)

    def _call_signature(self, *, tool_name: str, tool_args: Mapping[str, Any]) -> str:
        try:
            normalized = json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            normalized = "<?>"
        return f"{tool_name}:{normalized}"

    def _call_mcp_tool(
        self,
        *,
        tool_name: str,
        tool_args: Mapping[str, Any],
        step_index: int,
        signature: Optional[str],
        results: list[dict[str, Any]],
    ) -> Literal["continue", "break"]:
        self._mcp_tool_ctx = McpToolContext(step_index=step_index, say="", signature=signature, results=results)

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

    def _workspace_info(self) -> dict[str, Any]:
        rois: list[dict[str, Any]] = []
        for r in self.workspace.rois:
            if not r.active:
                continue
            rois.append({"name": r.name, "description": r.description or ""})
        active_roi_names = {r["name"] for r in rois}
        anchors: list[dict[str, Any]] = []
        for a in self.workspace.anchors:
            if not a.active:
                continue
            anchors.append(
                {
                    "name": a.name,
                    "description": a.description or "",
                    "linked_rois": [r for r in (a.linked_rois or []) if r in active_roi_names],
                }
            )
        return {"workspace_path": str(self.workspace.source_path), "rois": rois, "anchors": anchors}

    def _memory_entries_for_llm(self) -> list[dict[str, Any]]:
        turns = int(self.config.memory_turns)
        if turns == -1:
            return list(self._memory)
        if turns <= 0:
            return []
        return list(self._memory[-turns:])

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _resolve_linked_rois(self, next_action: Mapping[str, Any]) -> list[str]:
        args = next_action.get("args", {})
        if isinstance(args, dict):
            anchor = args.get("anchor", None)
            if isinstance(anchor, str) and anchor.strip():
                try:
                    anchor_obj = self.workspace.anchor(anchor.strip())
                    if not anchor_obj.active:
                        return []
                    linked = list(anchor_obj.linked_rois or [])
                    return [x for x in linked if isinstance(x, str) and x.strip()]
                except Exception:
                    return []

        linked_rois = next_action.get("linked_rois", [])
        if not isinstance(linked_rois, list):
            return []
        out: list[str] = []
        for r in linked_rois:
            if not isinstance(r, str) or not r.strip():
                continue
            rr = r.strip()
            try:
                roi = self.workspace.roi(rr)
            except Exception:
                continue
            if not roi.active:
                continue
            out.append(rr)
        return out

    def _observe(self, *, rois: list[str]) -> Mapping[str, Any]:
        if not rois:
            return {"rois": [], "results": {}, "unreadable": []}

        self._raise_if_aborted()
        resolved = []
        for r in rois:
            rr = str(r).strip()
            if not rr:
                continue
            roi = self.workspace.roi(rr)
            if not roi.active:
                continue
            resolved.append(roi)
        if not resolved:
            return {"rois": [], "results": {}, "unreadable": []}
        roi_names = [r.name for r in resolved]
        roi_descs = [r.description or "" for r in resolved]

        last: Mapping[str, Any] = {"rois": roi_names, "results": {}, "unreadable": list(roi_names)}
        attempts = max(1, int(self.config.observe_max_retries))
        for attempt in range(1, attempts + 1):
            self._raise_if_aborted()
            images = [self.capturer.capture_roi(roi) for roi in resolved]
            roi_items = list(zip(roi_names, roi_descs, images))

            step = self.logger.start_step(f"observe_{attempt}_{roi_names[0] if roi_names else 'none'}")
            for name, _desc, img in roi_items:
                step.save_image(f"roi_{name}.png", img)

            self._raise_if_aborted()
            extracted = self.llm_tool.observe_rois(roi_items=roi_items)
            self._accumulate_last_usage(self.llm_tool)
            self._raise_if_aborted()

            results = extracted.get("results", {})
            unreadable = extracted.get("unreadable", [])
            if not isinstance(results, dict):
                results = {}
            if not isinstance(unreadable, list):
                unreadable = []

            unreadable_s = [str(x).strip() for x in unreadable if str(x).strip()]
            unreadable_s = [x for x in unreadable_s if x in roi_names]

            norm_results: dict[str, Any] = {}
            for n in roi_names:
                item = results.get(n, None)
                if isinstance(item, dict):
                    val = item.get("value", None)
                    notes = item.get("notes", None)
                    conf = item.get("confidence", None)
                    norm_results[n] = {
                        "value": None if val is None else str(val),
                        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                        "notes": "" if notes is None else str(notes),
                    }
                else:
                    norm_results[n] = {"value": None, "confidence": None, "notes": ""}

            last = {"rois": roi_names, "results": norm_results, "unreadable": unreadable_s}
            step.write_meta({"tool": "observe_internal", "attempt": attempt, "rois": roi_names, "llm": dict(extracted)})

            if not unreadable_s:
                return last

        bad = list(last.get("unreadable", []) or [])
        if bad:
            raise RuntimeError(
                f"ROI(s) unreadable for {attempts} consecutive attempts: {', '.join(bad)}. "
                "ROI likely miscalibrated; run /calibration_tool and retry."
            )
        return last

    def _update_memory(
        self,
        *,
        user_command: str,
        last_action_done: Optional[Mapping[str, Any]],
        observation_of_last_action: Optional[Mapping[str, Any]],
        tools: list[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._raise_if_aborted()
        out = self.llm_agent.update_memory_step(
            system_prompt=SYSTEM_PROMPT,
            user_command=user_command,
            plan_text=self._plan_text,
            memory_entries=self._memory_entries_for_llm(),
            last_action_done=last_action_done,
            observation_of_last_action=observation_of_last_action,
            workspace_info=dict(workspace_info),
            tools=list(tools),
        )
        self._accumulate_last_usage(self.llm_agent)
        self._raise_if_aborted()

        rationale = str(out.get("rationale", "")).strip()
        next_action = out.get("next_action", None)
        if not rationale:
            raise RuntimeError("update_memory_step returned empty rationale.")
        if not isinstance(next_action, dict):
            raise RuntimeError("update_memory_step returned invalid next_action.")

        tool_name = str(next_action.get("tool", "")).strip()
        args = next_action.get("args", {})
        linked_rois = next_action.get("linked_rois", None)
        if not tool_name:
            raise RuntimeError("update_memory_step returned empty next_action.tool.")
        if not isinstance(args, dict):
            args = {}
        if not isinstance(linked_rois, list) or not all(isinstance(x, str) for x in linked_rois):
            raise RuntimeError("update_memory_step next_action.linked_rois must be list[str].")

        resolved_linked = self._resolve_linked_rois({"tool": tool_name, "args": args, "linked_rois": linked_rois})
        return {"rationale": rationale, "next_action": {"tool": tool_name, "args": dict(args), "linked_rois": resolved_linked}}

    def _tool_schemas_text(self, *, openai_tools: list[Mapping[str, Any]]) -> str:
        parts: list[str] = []
        for t in openai_tools:
            name = str(t.get("name", "")).strip()
            desc = str(t.get("description", "")).strip()
            params = t.get("parameters", {})
            try:
                schema = json.dumps(params, ensure_ascii=False)
            except Exception:
                schema = "{}"
            if name:
                parts.append(f"- {name}: {desc}\n  inputSchema: {schema}")
        return "\n".join(parts) if parts else "(none)"

    def _memory_for_run(self, *, run_index: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self._memory:
            if not isinstance(e, dict):
                continue
            try:
                if int(e.get("run_index", -1)) != int(run_index):
                    continue
            except Exception:
                continue
            out.append(e)
        return out

    def _tool_results_for_llm(self, *, tool_results: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """
        Produce a compact, JSON-safe representation of tool results for LLM consumption.
        """

        def compact(v: Any) -> Any:
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, list):
                if len(v) > 20:
                    return [compact(x) for x in v[:20]] + ["...(truncated)"]
                return [compact(x) for x in v]
            if isinstance(v, dict):
                out: dict[str, Any] = {}
                for k, vv in v.items():
                    if len(out) >= 30:
                        out["...(truncated)"] = True
                        break
                    if not isinstance(k, str):
                        continue
                    # Drop very large blobs/paths if present; keep top-level signals.
                    if k in {"images", "last_images", "image", "screenshot", "raw", "data_url"}:
                        continue
                    out[k] = compact(vv)
                return out
            return str(v)

        out: list[Mapping[str, Any]] = []
        for item in tool_results or []:
            if not isinstance(item, Mapping):
                continue
            tool = item.get("action", item.get("tool", None))
            action_input = item.get("action_input", item.get("args", None))
            result = item.get("result", None)
            say = item.get("say", None)
            out.append(
                {
                    "tool": None if tool is None else str(tool),
                    "args": compact(action_input),
                    "result": compact(result),
                    "say": None if say is None else str(say),
                }
            )
        return out

    def _memory_token_estimate(self) -> int:
        try:
            s = json.dumps(list(self._memory), ensure_ascii=False, default=str)
        except Exception:
            s = str(self._memory)
        return int(estimate_tokens(text=s, model=str(self.config.agent_model)))

    def compress_memory(self, *, reason: str, tool_results_by_run: Optional[Mapping[int, list[Mapping[str, Any]]]] = None) -> Mapping[str, Any]:
        """
        Compress in-memory session history by runs, moving details to `archive_memory`.
        Emits start/finish events for the TUI.
        """
        before_tokens = self._memory_token_estimate()
        self._emit(
            "memory_compress",
            phase="start",
            reason=str(reason or "").strip() or "(none)",
            before_tokens=int(before_tokens),
            threshold_tokens=int(getattr(self.config, "memory_compress_threshold_tokens", 0)),
        )
        try:
            out = self.llm_agent.compression_memory(
                system_prompt=SYSTEM_PROMPT,
                memory_entries=list(self._memory),
                archive_memory_entries=list(self._archive_memory),
                tool_results_by_run=tool_results_by_run or {},
            )
            self._accumulate_last_usage(self.llm_agent)
        except Exception as e:
            self._emit(
                "memory_compress",
                phase="error",
                reason=str(reason or "").strip() or "(none)",
                before_tokens=int(before_tokens),
                error=str(e),
            )
            raise
        mem_new = out.get("memory", None)
        arch_new = out.get("archive_memory", None)
        if isinstance(mem_new, list) and all(isinstance(x, Mapping) for x in mem_new):
            self._memory = [dict(x) for x in mem_new]  # type: ignore[arg-type]
        if isinstance(arch_new, list) and all(isinstance(x, Mapping) for x in arch_new):
            self._archive_memory = [dict(x) for x in arch_new]  # type: ignore[arg-type]
        after_tokens = self._memory_token_estimate()
        self._emit(
            "memory_compress",
            phase="done",
            reason=str(reason or "").strip() or "(none)",
            before_tokens=int(before_tokens),
            after_tokens=int(after_tokens),
            archive_entries=int(len(self._archive_memory)),
            memory_entries=int(len(self._memory)),
        )
        return {
            "before_tokens": int(before_tokens),
            "after_tokens": int(after_tokens),
            "archive_entries": int(len(self._archive_memory)),
            "memory_entries": int(len(self._memory)),
        }

    def Chat(
        self,
        *,
        user_command: str,
        run_index: int,
        openai_tools: list[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        tools_text = self._tool_schemas_text(openai_tools=openai_tools)
        self._raise_if_aborted()
        reply_obj = self.llm_agent.chat_reply(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_command,
            memory_entries=list(self._memory),
            workspace_info=dict(workspace_info),
            tools_text=tools_text,
        )
        self._accumulate_last_usage(self.llm_agent)
        self._raise_if_aborted()
        reply = str(reply_obj.get("reply", "")).strip()
        if not reply:
            reply = "(no reply)"

        self._memory.append(
            {
                "t": self._now_iso(),
                "run_index": int(run_index),
                "mode": "chat",
                "user_message": user_command,
                "assistant_reply": reply,
            }
        )
        self._emit("chat", text=reply)
        return [{"mode": "chat", "reply": reply}]

    def ReAct(
        self,
        *,
        user_command: str,
        run_index: int,
        openai_tools: list[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], bool]:
        results: list[dict[str, Any]] = []
        known_tools = {str(t.get("name", "")).strip() for t in openai_tools if str(t.get("name", "")).strip()}

        try:
            self._raise_if_aborted()
            plan_out = self.llm_agent.plan_step(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_command,
                memory_entries=self._memory_entries_for_llm(),
                workspace_info=dict(workspace_info),
                tools_text=self._tool_schemas_text(openai_tools=openai_tools),
            )
            self._accumulate_last_usage(self.llm_agent)
            self._raise_if_aborted()
            self._plan_text = str(plan_out.get("plan_text", "")).strip() or "(none)"
        except Exception:
            self._plan_text = "(none)"

        self._emit("plan_text", text=self._plan_text)

        finished = False
        self._raise_if_aborted()
        seed = self._update_memory(
            user_command=user_command,
            last_action_done=None,
            observation_of_last_action=None,
            tools=list(openai_tools),
            workspace_info=dict(workspace_info),
        )
        self._memory.append(
            {
                "t": self._now_iso(),
                "run_index": int(run_index),
                "mode": "agent",
                "user_message": user_command,
                "plan_text": self._plan_text,
                "last_action_done": None,
                "observation_of_last_action": None,
                "rationale": str(seed["rationale"]),
                "next_action": dict(seed["next_action"]),
            }
        )

        for step_index in range(1, int(self.config.max_steps) + 1):
            self._raise_if_aborted()
            current = self._memory[-1]
            next_action = current.get("next_action", {})
            if not isinstance(next_action, dict):
                raise RuntimeError("Missing next_action in memory.")

            tool_name = str(next_action.get("tool", "")).strip()
            tool_args = next_action.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            if tool_name not in known_tools:
                raise RuntimeError(f"Unknown tool requested: {tool_name!r}")

            self._raise_if_aborted()
            signature = self._call_signature(tool_name=tool_name, tool_args=tool_args)
            flow = self._call_mcp_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                step_index=step_index,
                signature=signature,
                results=results,
            )
            self._raise_if_aborted()

            last_action_done: dict[str, Any] = {
                "tool": tool_name,
                "args": dict(tool_args),
                "status": "finished",
                "error": None,
            }

            observed_rois = self._resolve_linked_rois(next_action)
            observation_of_last_action: Optional[Mapping[str, Any]] = None
            if observed_rois:
                observation_of_last_action = self._observe(rois=observed_rois)

            self._raise_if_aborted()
            out = self._update_memory(
                user_command=user_command,
                last_action_done=last_action_done,
                observation_of_last_action=observation_of_last_action,
                tools=list(openai_tools),
                workspace_info=dict(workspace_info),
            )

            entry = {
                "t": self._now_iso(),
                "run_index": int(run_index),
                "mode": "agent",
                "last_action_done": last_action_done,
                "observation_of_last_action": observation_of_last_action,
                "rationale": str(out["rationale"]),
                "next_action": dict(out["next_action"]),
            }
            self._memory.append(entry)

            self._raise_if_aborted()
            narration = self.llm_agent.narrate_react_step(
                system_prompt=SYSTEM_PROMPT,
                last_action_done=last_action_done,
                observation_of_last_action=observation_of_last_action,
                rationale=str(out["rationale"]),
                next_action=dict(out["next_action"]),
            )
            self._accumulate_last_usage(self.llm_agent)
            self._raise_if_aborted()

            if not isinstance(narration, dict):
                narration = {}

            self._emit(
                "step_blocks",
                step=step_index,
                action=str(narration.get("action", "")).strip(),
                observe=str(narration.get("observe", "")).strip(),
                think=str(narration.get("think", "")).strip(),
                next=str(narration.get("next", "")).strip(),
                raw=entry,
            )

            self._last_action_log = f"{tool_name}({tool_args})"

            if flow == "break":
                finished = (tool_name == "finish")
                break
            if str(out["next_action"].get("tool", "")).strip() == "finish":
                finished = True
                break

        return results, finished

    def run(self, *, user_command: str) -> list[Mapping[str, Any]]:
        user_command = str(user_command or "").strip()
        if not user_command:
            return []

        # Reset abort flag for a new run so a previous Ctrl+C doesn't permanently poison future runs.
        self._clear_abort()

        # Reload workspace on each run so chat sessions track user edits (add/delete ROIs/anchors).
        try:
            self.workspace = load_workspace(self.workspace.source_path)
        except Exception:
            pass

        openai_tools = self._openai_tools()
        workspace_info = self._workspace_info()

        mode = str(self.config.run_mode)
        if mode == "auto":
            try:
                self._raise_if_aborted()
                classified = self.llm_agent.classify_run_mode(system_prompt=SYSTEM_PROMPT, user_message=user_command)
                self._accumulate_last_usage(self.llm_agent)
                self._raise_if_aborted()
                m = str(classified.get("mode", "")).strip().lower()
                if m in {"agent", "chat"}:
                    mode = m
            except Exception:
                mode = "agent"

        last_mode: Optional[str] = None
        if self._memory:
            try:
                last_mode = str(self._memory[-1].get("mode", "")).strip().lower()  # type: ignore[union-attr]
            except Exception:
                last_mode = None

        should_bump = (mode == "agent") or (mode == "chat" and last_mode != "chat")
        if should_bump:
            self._run_index += 1
        run_index = int(self._run_index)

        self.logger.narrate(f"Workspace: {self.workspace.source_path}")
        self.logger.narrate(
            f"Mode: {mode} (agent_model={self.config.agent_model}, tool_call_model={self.config.tool_call_model})"
        )
        self.logger.narrate(f"Run index: {run_index}")

        if mode == "chat":
            out = self.Chat(user_command=user_command, run_index=run_index, openai_tools=openai_tools, workspace_info=workspace_info)
            # Auto-compress after chat turns if memory grows too large.
            thr = int(getattr(self.config, "memory_compress_threshold_tokens", 0))
            if thr > 0:
                try:
                    if self._memory_token_estimate() >= thr:
                        self.compress_memory(reason=f"auto_threshold_reached({thr})")
                except Exception:
                    pass
            return out

        results, finished = self.ReAct(
            user_command=user_command, run_index=run_index, openai_tools=openai_tools, workspace_info=workspace_info
        )

        if finished:
            run_mem = [
                e
                for e in self._memory_for_run(run_index=run_index)
                if isinstance(e, dict)
                and e.get("mode") == "agent"
                and "summary_of_run" not in e
                and "summary_of_last_run" not in e
            ]
            summary_obj = self.llm_agent.summarize_run(
                system_prompt=SYSTEM_PROMPT,
                mode="agent",
                plan_text=self._plan_text,
                memory_entries=run_mem,
                tool_results=self._tool_results_for_llm(tool_results=results),
            )
            self._accumulate_last_usage(self.llm_agent)
            summary = str(summary_obj.get("summary", "")).strip() or "Finished."
            self._memory.append(
                {
                    "t": self._now_iso(),
                    "run_index": int(run_index),
                    "mode": "agent",
                    "summary_of_run": summary,
                }
            )
            self._emit("finish", say=summary)

        # Auto-compress after agent runs if memory grows too large.
        thr = int(getattr(self.config, "memory_compress_threshold_tokens", 0))
        if thr > 0:
            try:
                if self._memory_token_estimate() >= thr:
                    self.compress_memory(reason=f"auto_threshold_reached({thr})")
            except Exception:
                pass

        self.logger.narrate(f"Logs: {self.logger.run_root}")
        return results

    def chat_cli(self, *, first_message: str | None = None) -> None:
        self.logger.narrate("CLI chat: type a command, or 'exit' to quit.")

        def run_one(text: str) -> None:
            try:
                self.run(user_command=text)
            except KeyboardInterrupt:
                self.logger.narrate("Aborted.")
            except Exception as e:
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
