from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from PIL import Image
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall


def _img_to_data_url(img: Image.Image) -> str:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_json_object(text: str) -> Mapping[str, Any]:
    # Be tolerant: find the first {...} block and parse it.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    snippet = text[start : end + 1]
    obj = json.loads(snippet)
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

    def react_step(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        memory_text: str,
        plan_text: str,
        observation_text: str,
        observation_images: Iterable[tuple[str, str, Image.Image]],
        tools: Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        images_payload: list[dict[str, Any]] = []
        for name, description, img in observation_images:
            desc = (description or "").strip()
            label = f"Image: {name}" if not desc else f"Image: {name}\nROI description: {desc}"
            images_payload.append({"type": "input_text", "text": label})
            images_payload.append({"type": "input_image", "image_url": _img_to_data_url(img)})

        full_user = (
            f"{user_prompt}\n\n"
            f"SESSION MEMORY (recent):\n{memory_text}\n\n"
            f"PLAN:\n{plan_text}\n\n"
            f"OBSERVATION (text):\n{observation_text}\n\n"
            "Call exactly one tool. Keep any narration short and demo-friendly."
        )

        resp = self._client.responses.create(
            model=self._config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "input_text", "text": full_user}, *images_payload]},
            ],
            tools=list(tools),
            tool_choice="required",
            parallel_tool_calls=False,
            timeout=self._config.timeout_s,
        )
        self._set_last_usage(resp)
        out_text = getattr(resp, "output_text", None) or ""

        tool_calls: list[dict[str, Any]] = []
        for item in getattr(resp, "output", None) or []:
            if isinstance(item, ResponseFunctionToolCall) and item.name:
                args_raw = item.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else {}
                except Exception:
                    args = {}
                tool_calls.append({"name": item.name, "arguments": args, "call_id": item.call_id})

        return {"text": out_text, "tool_calls": tool_calls}

    def plan_step(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        memory_text: str,
        workspace_text: str,
    ) -> Mapping[str, Any]:
        planner_system = (
            f"{system_prompt}\n\n"
            "PLANNING MODE:\n"
            "- Do not call any tools.\n"
            "- Return ONLY one JSON object.\n"
            "- JSON keys: plan (list of short steps), success_criteria (list), risks (list).\n"
        )
        full_user = (
            f"USER COMMAND: {user_prompt}\n\n"
            f"SESSION MEMORY (recent):\n{memory_text}\n\n"
            f"WORKSPACE:\n{workspace_text}\n\n"
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

