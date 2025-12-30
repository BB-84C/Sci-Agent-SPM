from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from PIL import Image


def _img_to_data_url(img: Image.Image) -> str:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_json_object(text: str) -> Mapping[str, Any]:
    # Be tolerant: find the first JSON object and ignore any trailing text/objects.
    start = text.find("{")
    if start == -1:
        raise ValueError("Model did not return a JSON object.")
    snippet = text[start:]
    obj, _end = json.JSONDecoder().raw_decode(snippet)
    if not isinstance(obj, dict):
        raise ValueError("Model JSON response must be an object.")
    return obj


def estimate_tokens(*, text: str, model: str = "gpt-5.2") -> int:
    """
    Best-effort token estimate for the given text.

    Uses `tiktoken` if available; otherwise falls back to a rough heuristic (~4 chars/token).
    """
    s = text or ""
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return int(len(enc.encode(s)))
    except Exception:
        # Conservative-ish fallback.
        return max(1, int(len(s) / 4))


def _try_extract_json_object(text: str) -> Optional[Mapping[str, Any]]:
    try:
        return _extract_json_object(text)
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class LlmConfig:
    model: str = "gpt-5.2"
    timeout_s: float = 60.0


class OpenAiMultimodalClient:
    """
    Minimal OpenAI client wrapper.
    Reads API key from env var OPENAI_API (per user request); also falls back to OPENAI_API_KEY.
    """

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self.last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OpenAI API key: set env var OPENAI_API (or OPENAI_API_KEY).")

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'openai' for agent mode.\n"
                f"Python: {sys.executable}\n"
                "Fix: python -m pip install -r requirements.txt\n"
                "Tip: ensure you're using the same Python/venv you run the program with."
            ) from e

        self._client = OpenAI(api_key=key)

    def _set_last_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return

        def get_int(obj: Any, key: str) -> int:
            v = None
            if isinstance(obj, dict):
                v = obj.get(key, None)
            else:
                v = getattr(obj, key, None)
            try:
                return int(v) if v is not None else 0
            except Exception:
                return 0

        in_tok = get_int(usage, "input_tokens")
        out_tok = get_int(usage, "output_tokens")
        total = get_int(usage, "total_tokens")
        if total <= 0:
            total = max(0, in_tok + out_tok)
        self.last_usage = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": total}

    def extract_readouts(
        self,
        *,
        roi_items: list[tuple[str, str, Image.Image]],
    ) -> Mapping[str, Any]:
        """
        Extract readout values from ROI screenshots.

        Returns a JSON dict:
          { "values": { "<roi_name>": "<value string or null>" }, "unreadable": ["<roi_name>"], "reason": "<short>" }
        """
        system = (
            "You extract UI readout values from ROI screenshots.\n"
            "Return ONLY one JSON object with exactly these keys:\n"
            '{ "values": { "<roi_name>": "<value string or null>" }, "unreadable": ["<roi_name>"], "reason": "<short>" }\n'
            "Rules:\n"
            "- If a value cannot be read confidently, set it to null and include the ROI name in unreadable.\n"
            "- Keep value strings concise and include units when visible (e.g., '700 mV', '50 pA').\n"
            "- No markdown. No code fences. No extra text."
        )
        lines = []
        for n, d, _img in roi_items:
            lines.append(f"- {n}: {d or '(none)'}")
        user = "Extract the current readout value(s) from these ROI images.\n\nROIs:\n" + ("\n".join(lines) if lines else "(none)")

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user}]
        for roi_name, _desc, img in roi_items:
            content.append({"type": "input_text", "text": f"ROI image: {roi_name}"})
            content.append({"type": "input_image", "image_url": _img_to_data_url(img)})

        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def observe_rois(
        self,
        *,
        roi_items: list[tuple[str, str, Image.Image]],
    ) -> Mapping[str, Any]:
        """
        Observe arbitrary ROIs and return structured values.

        Returns a JSON dict:
          {
            "results": { "<roi_name>": { "value": "<string or null>", "confidence": <float 0..1>, "notes": "<short>" } },
            "unreadable": ["<roi_name>"]
          }
        """
        system = (
            "You observe UI ROIs based on screenshots and ROI descriptions.\n"
            "Return ONLY one JSON object with exactly these keys:\n"
            '{ "results": { "<roi_name>": { "value": "<string or null>", "confidence": <float>, "notes": "<short>" } }, '
            '"unreadable": ["<roi_name>"] }\n'
            "Rules:\n"
            "- Use the ROI description as authoritative for semantics (e.g., discrete tokens like <scanning>/<idle>).\n"
            "- First, look at the image and state what you see; then map that to the value.\n"
            "- notes MUST start with a brief literal visual description beginning with \"I see ...\".\n"
            "  Then add \"Meaning: ...\" to explain how that maps to the chosen value.\n"
            "- If an ROI is unreadable, set value to null and include the ROI name in unreadable.\n"
            "- confidence should be between 0 and 1 when possible.\n"
            "- No markdown. No code fences. No extra text."
        )
        lines = []
        for n, d, _img in roi_items:
            lines.append(f"- {n}: {d or '(none)'}")
        user = (
            "Based on each ROI description and its image, what do you see?\n"
            "Then extract the current value(s) for each ROI.\n\n"
            "ROIs:\n" + ("\n".join(lines) if lines else "(none)")
        )

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user}]
        for roi_name, _desc, img in roi_items:
            content.append({"type": "input_text", "text": f"ROI image: {roi_name}"})
            content.append({"type": "input_image", "image_url": _img_to_data_url(img)})

        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def plan_step(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        memory_entries: list[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
        tools_text: str,
    ) -> Mapping[str, Any]:
        planner_system = (
            f"{system_prompt}\n\n"
            "PLANNING MODE:\n"
            "- Do not call any tools.\n"
            "- Return ONLY one JSON object.\n"
            '- JSON keys: "plan_text" only.\n'
            "- plan_text: a concise bullet list describing the intended overall workflow.\n"
            "- Do NOT include tool calls or per-step args.\n"
        )
        try:
            mem_json = json.dumps(list(memory_entries), ensure_ascii=False, default=str)
        except Exception:
            mem_json = "[]"
        try:
            ws_json = json.dumps(dict(workspace_info), ensure_ascii=False, default=str)
        except Exception:
            ws_json = "{}"
        full_user = (
            f"USER COMMAND: {user_prompt}\n\n"
            f"SESSION MEMORY (structured, recent):\n{mem_json}\n\n"
            f"TOOLS (schemas):\n{tools_text}\n\n"
            f"WORKSPACE (summary):\n{ws_json}\n\n"
            "Return ONLY valid JSON."
        )
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": planner_system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def update_memory_step(
        self,
        *,
        system_prompt: str,
        user_command: str,
        plan_text: str,
        memory_entries: list[Mapping[str, Any]],
        last_action_done: Optional[Mapping[str, Any]],
        observation_of_last_action: Optional[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
        tools: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """
        Think + Next stage. Returns a JSON dict:
          { "rationale": "<short>", "next_action": { "tool": "<name>", "args": {...}, "linked_rois": ["..."] } }
        """
        think_system = (
            f"{system_prompt}\n\n"
            "THINK+NEXT MODE:\n"
            "- You are the decision-maker. Pick the next MCP tool call.\n"
            "- Return ONLY one JSON object.\n"
            '- JSON keys: "rationale" (required), "next_action" (required).\n'
            '- next_action keys: "tool" (required), "args" (required object), "linked_rois" (required list of ROI names).\n'
            "- Use only tools that exist in TOOLS.\n"
            "- Use only anchors/ROIs that exist in WORKSPACE.\n"
            "- If you are done, set next_action.tool to \"finish\".\n"
        )
        try:
            mem_json = json.dumps(list(memory_entries), ensure_ascii=False, default=str)
        except Exception:
            mem_json = "[]"
        try:
            last_json = json.dumps(dict(last_action_done or {}), ensure_ascii=False, default=str)
        except Exception:
            last_json = "{}"
        try:
            obs_json = json.dumps(dict(observation_of_last_action or {}), ensure_ascii=False, default=str)
        except Exception:
            obs_json = "{}"
        try:
            ws_json = json.dumps(dict(workspace_info), ensure_ascii=False, default=str)
        except Exception:
            ws_json = "{}"
        try:
            tools_json = json.dumps(list(tools), ensure_ascii=False, default=str)
        except Exception:
            tools_json = "[]"

        full_user = (
            f"USER COMMAND: {user_command}\n\n"
            f"PLAN (constant):\n{plan_text}\n\n"
            f"MEMORY (structured, recent):\n{mem_json}\n\n"
            f"LAST_ACTION_DONE:\n{last_json}\n\n"
            f"OBSERVATION_OF_LAST_ACTION:\n{obs_json}\n\n"
            f"WORKSPACE (summary):\n{ws_json}\n\n"
            f"TOOLS (OpenAI function schemas):\n{tools_json}\n\n"
            "Return ONLY valid JSON."
        )
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": think_system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def narrate_react_step(
        self,
        *,
        system_prompt: str,
        last_action_done: Mapping[str, Any],
        observation_of_last_action: Optional[Mapping[str, Any]],
        rationale: str,
        next_action: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        UI narrator for one completed ReAct step.

        Returns JSON:
          { "action": "<text>", "observe": "<text>", "think": "<text>", "next": "<text>" }
        """
        narrator_system = (
            f"{system_prompt}\n\n"
            "NARRATION MODE (FORMATTER ONLY):\n"
            "- You are a formatter that converts the provided JSON fields into 4 short, human-friendly agent-style blocks.\n"
            "- Do NOT add new facts. Do NOT infer missing steps. Use ONLY the provided JSON.\n"
            "- Do NOT mention tool access, permissions, modes, screenshots, or limitations.\n"
            '  Never say phrases like: "I can\'t", "I cannot", "no access", "this turn", "provide screenshots", "re-enable".\n'
            "- Return ONLY one JSON object with exactly these keys: action, observe, think, next.\n"
            "- Each field must be 1-2 sentences.\n\n"
            "ACTION block rules:\n"
            "- Describe what was just done in natural first-person past tense.\n"
            '- If last_action_done.status == "finished": e.g., "I set bias_input to 400 mV."\n'
            '- If status != "finished" or error present: e.g., "The step failed while running <tool>(...). Error: <error>."\n'
            "OBSERVE block rules:\n"
            "- Report what the ROIs show in plain language.\n"
            "- If observation_of_last_action is null OR rois is empty: say you skipped verification because there were no linked ROIs.\n"
            "- Else: mention key ROI values; include confidence only if present and <0.8.\n"
            "THINK block rules:\n"
            "- Rephrase the provided rationale lightly as an agent intent, but do not introduce new reasoning.\n"
            "NEXT block rules:\n"
            "- State the next step as what you will do next, and what you will verify (using linked_rois if present).\n\n"
            "Example:\n"
            "{\n"
            "  \"action\": \"I entered 400 mV into the bias field.\",\n"
            "  \"observe\": \"The bias readout shows 0.400 V.\",\n"
            "  \"think\": \"Bias looks correct, so the next step is setting the current.\",\n"
            "  \"next\": \"Next I\u2019ll set the current to 10 pA and then confirm it on set_current_readout.\"\n"
            "}\n"
        )
        try:
            action_json = json.dumps(dict(last_action_done), ensure_ascii=False, default=str)
        except Exception:
            action_json = "{}"
        try:
            obs_json = json.dumps(dict(observation_of_last_action or {}), ensure_ascii=False, default=str)
        except Exception:
            obs_json = "{}"
        try:
            next_json = json.dumps(dict(next_action), ensure_ascii=False, default=str)
        except Exception:
            next_json = "{}"

        full_user = (
            f"ACTION (raw):\n{action_json}\n\n"
            f"OBSERVE (raw):\n{obs_json}\n\n"
            f"THINK (rationale):\n{rationale}\n\n"
            f"NEXT (raw):\n{next_json}\n\n"
            "Return ONLY valid JSON."
        )
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": narrator_system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            max_output_tokens=300,
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def assess_step_limit(
        self,
        *,
        system_prompt: str,
        user_command: str,
        plan_text: str,
        memory_text: str,
        observation_text: str,
        executed_steps_text: str,
        max_steps: int,
    ) -> Mapping[str, Any]:
        assessor_system = (
            f"{system_prompt}\n\n"
            "STEP LIMIT ASSESSMENT MODE:\n"
            "- The run hit the max step limit before a terminal tool was called.\n"
            "- Decide whether this looks like a loop/stall or simply a long task.\n"
            "- Return ONLY one JSON object.\n"
            '- JSON keys: "category" (required; one of "loop"|"too_long"|"unclear"), '
            '"progress" (required; short), "next" (required; short), "say" (required; 1-3 sentences to user).\n'
            "- Do not reveal hidden chain-of-thought.\n"
        )
        full_user = (
            f"USER COMMAND: {user_command}\n\n"
            f"MAX STEPS: {max_steps}\n\n"
            f"SESSION MEMORY (recent):\n{memory_text}\n\n"
            f"PLAN:\n{plan_text}\n\n"
            f"CURRENT OBSERVATION:\n{observation_text}\n\n"
            f"EXECUTED STEPS (recent):\n{executed_steps_text}\n\n"
            "Return ONLY valid JSON."
        )
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": assessor_system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            max_output_tokens=350,
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def classify_run_mode(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> Mapping[str, Any]:
        """
        Classify whether a user message should run in ReAct agent mode or chat mode.

        Returns JSON:
          { "mode": "agent"|"chat", "reason": "<short>" }
        """
        system = (
            f"{system_prompt}\n\n"
            "MODE CLASSIFIER:\n"
            "- Decide whether the user is asking to PERFORM an on-desktop task (agent) or asking a question / discussion (chat).\n"
            "- Return ONLY one JSON object.\n"
            '- JSON keys: "mode" (required; "agent" or "chat"), "reason" (required; short).\n'
            "- If the user asks to click/type/scan/wait/change values in the app, choose agent.\n"
            "- If the user asks to explain, analyze logs, or discuss behavior, choose chat.\n"
        )
        full_user = f"USER MESSAGE:\n{user_message}\n\nReturn ONLY valid JSON."
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            max_output_tokens=120,
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def chat_reply(
        self,
        *,
        system_prompt: str,
        user_message: str,
        memory_entries: list[Mapping[str, Any]],
        workspace_info: Mapping[str, Any],
        tools_text: str,
    ) -> Mapping[str, Any]:
        """
        Chat-only mode: respond with text; do not call tools.

        Returns JSON:
          { "reply": "<assistant text>" }
        """
        system = (
            f"{system_prompt}\n\n"
            "CHAT MODE:\n"
            "- You are chatting with the user. Do NOT call tools.\n"
            "- You may reference WORKSPACE and TOOLS to explain what the agent can do, but you cannot execute actions.\n"
            "- Return ONLY one JSON object.\n"
            '- JSON key: "reply" (required string).\n'
        )
        try:
            mem_json = json.dumps(list(memory_entries), ensure_ascii=False, default=str)
        except Exception:
            mem_json = "[]"
        try:
            ws_json = json.dumps(dict(workspace_info), ensure_ascii=False, default=str)
        except Exception:
            ws_json = "{}"
        full_user = (
            f"MEMORY (structured):\n{mem_json}\n\n"
            f"WORKSPACE (summary):\n{ws_json}\n\n"
            f"TOOLS (schemas):\n{tools_text}\n\n"
            f"USER MESSAGE:\n{user_message}\n\n"
            "Return ONLY valid JSON."
        )
        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}]},
            ],
            tool_choice="none",
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        text = getattr(resp, "output_text", None) or ""
        return _extract_json_object(text)

    def summarize_run(
        self,
        *,
        system_prompt: str,
        mode: str,
        plan_text: str | None = None,
        memory_entries: list[Mapping[str, Any]],
        tool_results: list[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        """
        Summarize one run (agent or chat).

        Returns JSON:
          { "summary": "<short>" }
        """
        m = str(mode or "").strip().lower()
        if m not in {"agent", "chat"}:
            m = "agent"

        if m == "agent":
            system = (
                f"{system_prompt}\n\n"
                "RUN SUMMARY MODE (AGENT):\n"
                "- Summarize what was done step-by-step, referencing tool calls/results when helpful.\n"
                "- Include any deviations from the PLAN and how they were handled.\n"
                "- Include a self-rating from 0 to 10.\n"
                "- Return ONLY one JSON object.\n"
                '- JSON key: "summary" (required string).\n'
                "- The summary string MUST start with: 'Overall performance rating of this run: X/10.'\n"
                "- Keep it concise but concrete; mention key observed states (e.g., <idle>, countdown=0s).\n"
            )
        else:
            system = (
                f"{system_prompt}\n\n"
                "RUN SUMMARY MODE (CHAT):\n"
                "- Summarize the conversation: key questions, answers, decisions, and any follow-ups.\n"
                "- Do NOT claim to have executed tools.\n"
                "- Return ONLY one JSON object.\n"
                '- JSON key: "summary" (required string).\n'
                "- Keep it concise and actionable (what was clarified / decided).\n"
            )
        try:
            mem_json = json.dumps(list(memory_entries), ensure_ascii=False, default=str)
        except Exception:
            mem_json = "[]"
        try:
            tools_json = json.dumps(list(tool_results or []), ensure_ascii=False, default=str)
        except Exception:
            tools_json = "[]"
        plan = str(plan_text or "").strip()
        full_user = (
            f"MODE: {m}\n\n"
            f"PLAN TEXT:\n{plan if plan else '(none)'}\n\n"
            f"TOOL RESULTS (structured):\n{tools_json}\n\n"
            f"RUN MEMORY (structured):\n{mem_json}\n\n"
            "Return ONLY valid JSON."
        )
        def call_once(*, sys_text: str, user_text: str, max_out: int) -> str:
            resp = self._client.responses.create(
                model=self._config.model,
                input=[
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
                ],
                tool_choice="none",
                max_output_tokens=max_out,
                timeout=self._config.timeout_s,
            )
            self._set_last_usage(resp)
            return getattr(resp, "output_text", None) or ""

        # Retry with a "repair" prompt if the model returns malformed JSON.
        last_text = call_once(sys_text=system, user_text=full_user, max_out=300)
        obj = _try_extract_json_object(last_text)
        if obj is not None:
            return obj

        repair_system = (
            "You are a JSON repair assistant.\n"
            "Given a previous model output that was supposed to be JSON, return ONLY a valid JSON object.\n"
            "No markdown. No code fences. No extra text.\n"
            'Return exactly: { "summary": "<string>" } and ensure all quotes/newlines are properly escaped.'
        )
        repair_user = (
            f"MODE: {m}\n\n"
            "Previous output (invalid JSON):\n"
            f"{last_text}\n\n"
            "Return ONLY valid JSON now."
        )
        last_text = call_once(sys_text=repair_system, user_text=repair_user, max_out=350)
        obj = _try_extract_json_object(last_text)
        if obj is not None:
            return obj

        # Final attempt: re-ask summarizer but with explicit JSON escaping guidance.
        stricter_user = (
            full_user
            + "\n\nIMPORTANT: Return valid JSON. Do not include unescaped newlines inside strings; use \\n if needed."
        )
        last_text = call_once(sys_text=system, user_text=stricter_user, max_out=350)
        obj = _try_extract_json_object(last_text)
        if obj is not None:
            return obj
        raise ValueError(f"summarize_run: model did not return valid JSON. Last output: {last_text[:500]!r}")

    def compression_memory(
        self,
        *,
        system_prompt: str,
        memory_entries: list[Mapping[str, Any]],
        archive_memory_entries: list[Mapping[str, Any]] | None = None,
        tool_results_by_run: Mapping[int, list[Mapping[str, Any]]] | None = None,
    ) -> Mapping[str, Any]:
        """
        Compress agent memory by runs (run_index). Keeps one summary per run in `memory`,
        and moves detailed entries to `archive_memory`.

        Returns JSON:
          { "memory": [...], "archive_memory": [...], "summaries": { "<run_index>": "<summary>" } }
        """

        def to_int(x: Any, default: int) -> int:
            try:
                if isinstance(x, bool):
                    return default
                return int(x)
            except Exception:
                return default

        existing_archive: list[Mapping[str, Any]] = []
        if isinstance(archive_memory_entries, list):
            existing_archive = [e for e in archive_memory_entries if isinstance(e, Mapping)]

        groups: dict[int, list[Mapping[str, Any]]] = {}
        passthrough: list[Mapping[str, Any]] = []
        for e in memory_entries or []:
            if not isinstance(e, Mapping):
                continue
            if "run_index" not in e:
                passthrough.append(e)
                continue
            ridx = to_int(e.get("run_index"), -1)
            if ridx < 0:
                passthrough.append(e)
                continue
            groups.setdefault(ridx, []).append(e)

        compressed: list[Mapping[str, Any]] = list(passthrough)
        archive: list[Mapping[str, Any]] = list(existing_archive)
        summaries: dict[str, str] = {}

        for run_index in sorted(groups.keys()):
            entries = groups[run_index]
            run_mode = "chat"
            for e in entries:
                m = str(e.get("mode", "")).strip().lower()
                if m == "agent":
                    run_mode = "agent"
                    break

            # If a run is already compressed (only summary entries remain), skip it to avoid
            # duplicating archive entries indefinitely.
            has_non_summary = False
            for e in entries:
                if not isinstance(e, Mapping):
                    continue
                if "summary_of_run" in e or "summary_of_last_run" in e:
                    continue
                has_non_summary = True
                break
            if not has_non_summary:
                continue

            existing_summary: str | None = None
            for e in entries:
                if not isinstance(e, Mapping):
                    continue
                v = e.get("summary_of_run", None)
                if isinstance(v, str) and v.strip():
                    existing_summary = v.strip()
                    break
                v = e.get("summary_of_last_run", None)
                if isinstance(v, str) and v.strip():
                    existing_summary = v.strip()
                    break

            summary = existing_summary
            if summary is None:
                plan_text = ""
                for e in entries:
                    v = e.get("plan_text", None)
                    if isinstance(v, str) and v.strip():
                        plan_text = v.strip()
                        break
                tools_for_run: list[Mapping[str, Any]] = []
                if tool_results_by_run is not None:
                    try:
                        tools_for_run = list(tool_results_by_run.get(int(run_index), []) or [])
                    except Exception:
                        tools_for_run = []
                try:
                    out = self.summarize_run(
                        system_prompt=system_prompt,
                        mode=run_mode,
                        plan_text=plan_text,
                        memory_entries=list(entries),
                        tool_results=tools_for_run,
                    )
                    summary = str(out.get("summary", "")).strip() or "Finished."
                except Exception as e:
                    # Never fail compression due to summary JSON issues; fall back to a minimal summary.
                    if run_mode == "chat":
                        msgs: list[str] = []
                        for it in entries:
                            um = it.get("user_message", None)
                            ar = it.get("assistant_reply", None)
                            if isinstance(um, str) and um.strip():
                                msgs.append(f"Q: {um.strip()}")
                            if isinstance(ar, str) and ar.strip():
                                msgs.append(f"A: {ar.strip()}")
                        snippet = " | ".join(msgs[-6:])
                        summary = f"Chat run compressed (summary fallback due to error: {e}). {snippet}".strip()
                    else:
                        summary = f"Agent run compressed (summary fallback due to error: {e}).".strip()

            summaries[str(int(run_index))] = summary
            # Move original entries to archive, then keep only one summary entry in compressed memory.
            archive.extend(entries)
            compressed.append(
                {
                    "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "run_index": int(run_index),
                    "mode": run_mode,
                    "summary_of_run": summary,
                }
            )

        return {"memory": compressed, "archive_memory": archive, "summaries": summaries}


def compression_memory(
    *,
    client: OpenAiMultimodalClient,
    system_prompt: str,
    memory_entries: list[Mapping[str, Any]],
    archive_memory_entries: list[Mapping[str, Any]] | None = None,
    tool_results_by_run: Mapping[int, list[Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    """
    Convenience wrapper around `OpenAiMultimodalClient.compression_memory(...)`.
    """
    return client.compression_memory(
        system_prompt=system_prompt,
        memory_entries=memory_entries,
        archive_memory_entries=archive_memory_entries,
        tool_results_by_run=tool_results_by_run,
    )

