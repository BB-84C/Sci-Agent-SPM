from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
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
            "- If an ROI is unreadable, set value to null and include it in unreadable.\n"
            "- confidence should be between 0 and 1 when possible.\n"
            "- No markdown. No code fences. No extra text."
        )
        lines = []
        for n, d, _img in roi_items:
            lines.append(f"- {n}: {d or '(none)'}")
        user = "Observe and extract the current value(s) from these ROI images.\n\nROIs:\n" + ("\n".join(lines) if lines else "(none)")

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

