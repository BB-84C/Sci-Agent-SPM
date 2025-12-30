from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional, TYPE_CHECKING, Literal

from ..llm_client import _extract_json_object, _img_to_data_url

if TYPE_CHECKING:
    from ..agent import VisualAutomationAgent


_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MAX_TOTAL_SECONDS = 6 * 60 * 60
_SLEEP_CAP_SECONDS = 6 * 60 * 60
_FIRST_SLEEP_BUFFER_SECONDS = 10.0
_MAX_FIRST_SLEEP_SECONDS = 10 * 60.0


def _coerce_roi_names(action_input: Mapping[str, Any]) -> list[str]:
    """
    Back-compat: support either {"roi": "..."} or {"rois": ["...","..."]}.
    """
    rois = action_input.get("rois", None)
    if isinstance(rois, list):
        names = [str(x).strip() for x in rois if isinstance(x, (str, int, float)) and str(x).strip()]
        if names:
            return names
    roi = action_input.get("roi", None)
    if isinstance(roi, str) and roi.strip():
        return [roi.strip()]
    raise ValueError("wait_until requires action_input.roi (str) or action_input.rois (list[str])")


def _require_nonneg_seconds(action_input: Mapping[str, Any], key: str) -> float:
    value = action_input.get(key, None)
    if value is None:
        raise ValueError(f"wait_until requires action_input.{key}")
    if not isinstance(value, (int, float)):
        raise ValueError(f"wait_until requires numeric action_input.{key}")
    seconds = float(value)
    if seconds < 0:
        raise ValueError(f"wait_until requires non-negative action_input.{key}")
    return seconds


def _optional_pos_int(action_input: Mapping[str, Any], key: str, default: int) -> int:
    value = action_input.get(key, None)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"wait_until requires integer action_input.{key}")
    v = int(value)
    if v <= 0:
        raise ValueError(f"wait_until requires positive action_input.{key}")
    return v


def _coerce_done(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    raise ValueError(f"Model JSON must include boolean 'done', got: {value!r}")


def _coerce_sleep_seconds(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v > 0 else None
    if isinstance(value, str):
        try:
            v = float(value.strip())
            return v if v > 0 else None
        except Exception:
            return None
    return None


def _parse_wait_logic(action_input: Mapping[str, Any]) -> dict[str, Any]:
    """
    wait_logic is a JSON string.

    Required keys:
    - action_to_wait: string
    - reason: string
    """
    raw = action_input.get("wait_logic", None)
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
        except Exception as e:
            raise ValueError(f"wait_until.wait_logic must be valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("wait_until.wait_logic must be a JSON object.")
        logic = dict(obj)
    else:
        logic = {}

    action_to_wait = logic.get("action_to_wait", None)
    if not isinstance(action_to_wait, str) or not action_to_wait.strip():
        # Small tolerance for earlier drafts.
        action_to_wait = logic.get("action_to_wait_for", None)
        if isinstance(action_to_wait, str) and action_to_wait.strip():
            logic["action_to_wait"] = action_to_wait.strip()
        else:
            raise ValueError("wait_until requires wait_logic JSON with key action_to_wait (non-empty string).")

    reason = logic.get("reason", None)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("wait_until requires wait_logic JSON with key reason (non-empty string).")
    return logic


def _llm_choose_sleep(
    agent: "VisualAutomationAgent",
    *,
    roi_names: list[str],
    roi_descriptions: list[str],
    wait_logic: Mapping[str, Any],
    images: list[tuple[str, Any]],
) -> Mapping[str, Any]:
    system = (
        "You choose how long to wait by inspecting UI ROI screenshots.\n"
        "You are given a JSON wait_logic that defines what we are waiting for and why.\n"
        "Return ONLY one JSON object with exactly these keys:\n"
        '{ "sleep_seconds": <number>, "reason": "<short>" }\n'
        "Rules:\n"
        "- reason MUST start with a brief literal visual description beginning with \"I see ...\".\n"
        "  Then add \"Therefore: ...\" explaining how the visuals map to sleep_seconds.\n"
        "- If a countdown/remaining time is visible in ANY ROI, you MUST set sleep_seconds to the remaining time in seconds (WITHOUT buffer).\n"
        "- Supported time formats include e.g. '26.1s', '26 s', '0:26', '00:26', '1:02', '00:01:02'. Convert to seconds.\n"
        "- If no countdown/remaining time is visible, then based on general knowledge and common logic,\n"
        "  choose a reasonable time to wait before re-checking (the caller will observe again after waiting).\n"
        f"- Keep sleep_seconds <= {_MAX_FIRST_SLEEP_SECONDS:.0f}.\n"
        "- sleep_seconds must be > 0.\n"
        "No markdown. No code fences. No extra text."
    )
    roi_lines = []
    for n, d in zip(roi_names, roi_descriptions):
        roi_lines.append(f"- {n}: {d or '(none)'}")
    roi_block = "\n".join(roi_lines) if roi_lines else "(none)"
    try:
        logic_json = json.dumps(dict(wait_logic), ensure_ascii=False)
    except Exception:
        logic_json = "{}"
    user = (
        "Based on the ROI descriptions and images, what do you see?\n"
        "Then choose sleep_seconds for a single wait; the runtime will re-check ROIs after waiting.\n\n"
        f"wait_logic (JSON):\n{logic_json}\n\n"
        f"ROIs:\n{roi_block}\n"
    )

    content: list[Mapping[str, Any]] = [{"type": "input_text", "text": user}]
    for roi_name, img in images:
        content.append({"type": "input_text", "text": f"ROI image: {roi_name}"})
        content.append({"type": "input_image", "image_url": _img_to_data_url(img)})

    resp = agent.llm._client.responses.create(  # type: ignore[attr-defined]
        model=agent.llm._config.model,  # type: ignore[attr-defined]
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": content,
            },
        ],
        timeout=agent.llm._config.timeout_s,  # type: ignore[attr-defined]
    )
    agent.llm._set_last_usage(resp)  # type: ignore[attr-defined]
    agent._accumulate_last_usage()

    text = getattr(resp, "output_text", None) or ""
    obj = _extract_json_object(text)
    sleep_seconds = _coerce_sleep_seconds(obj.get("sleep_seconds", None))
    if sleep_seconds is None:
        raise ValueError("Model JSON must include positive numeric sleep_seconds.")
    sleep_seconds = min(float(sleep_seconds), float(_MAX_FIRST_SLEEP_SECONDS))
    if sleep_seconds <= 0:
        raise ValueError("sleep_seconds must be > 0.")
    return {"sleep_seconds": float(sleep_seconds), "reason": str(obj.get("reason", "")).strip()}


def handle(
    agent: "VisualAutomationAgent",
    *,
    step_index: int,
    action_input: Mapping[str, Any],
    say: str,
    signature: Optional[str],
    results: list[Mapping[str, Any]],
) -> Literal["continue", "break"]:
    agent._consecutive_observes = 0

    roi_names = _coerce_roi_names(action_input)
    seconds = _require_nonneg_seconds(action_input, "seconds")
    _max_rounds = _optional_pos_int(action_input, "max_rounds", _DEFAULT_MAX_ROUNDS)
    max_total_seconds = _require_nonneg_seconds(
        {"max_total_seconds": action_input.get("max_total_seconds", _DEFAULT_MAX_TOTAL_SECONDS)},
        "max_total_seconds",
    )
    wait_logic = _parse_wait_logic(action_input)

    resolved: list[Any] = []
    missing: list[str] = []
    for name in roi_names:
        try:
            resolved.append(agent.workspace.roi(name))
        except KeyError:
            missing.append(name)
    if missing:
        step = agent.logger.start_step("wait_until_invalid")
        meta = {"tool": "wait_until", "reason": "invalid_roi", "rois": roi_names, "missing": missing}
        step.write_meta(meta)
        result = {
            "reason": "wait_until_invalid_roi",
            "rois": roi_names,
            "missing": missing,
            "log_root": str(agent.logger.run_root),
        }
        results.append({"action": "wait_until", "action_input": dict(action_input), "result": result, "say": say})
        agent._emit("result", step=step_index, action="wait_until", result=result)
        return "continue"

    roi_descriptions = [getattr(roi, "description", "") or "" for roi in resolved]

    agent.logger.narrate(f"[WaitUntil] Waiting on ROI(s): {', '.join(roi_names)}.")

    roi_label = roi_names[0] if len(roi_names) == 1 else f"{roi_names[0]}+{len(roi_names) - 1}"
    step = agent.logger.start_step(f"wait_until_{roi_label}_sleep")
    last_step_rel = str(step.path.relative_to(agent.logger.run_root))

    captured: list[tuple[str, Any]] = []
    last_images: dict[str, str] = {}
    for roi_name, roi in zip(roi_names, resolved):
        img = agent.capturer.capture_roi(roi)
        img_name = f"before_{roi_name}.png"
        step.save_image(img_name, img)
        rel = str((step.path / img_name).relative_to(agent.logger.run_root))
        last_images[roi_name] = rel
        captured.append((roi_name, img))
    last_image_rel = last_images.get(roi_names[0], "")

    llm_error = ""
    try:
        choice = _llm_choose_sleep(
            agent,
            roi_names=roi_names,
            roi_descriptions=roi_descriptions,
            wait_logic=wait_logic,
            images=captured,
        )
    except Exception as e:
        llm_error = str(e)
        # Fallback: use caller-provided seconds if the sleep chooser fails.
        choice = {
            "sleep_seconds": float(seconds),
            "reason": "I see no reliable countdown; therefore using the requested seconds as fallback.",
        }

    sleep_hint = float(choice.get("sleep_seconds", seconds))
    sleep_s = min(sleep_hint + float(_FIRST_SLEEP_BUFFER_SECONDS), float(_MAX_FIRST_SLEEP_SECONDS))
    sleep_s = min(float(sleep_s), float(max_total_seconds), float(_SLEEP_CAP_SECONDS))
    if sleep_s < 0:
        sleep_s = 0.0

    meta: dict[str, Any] = {
        "tool": "wait_until",
        "mode": "one_shot_sleep",
        "rois": roi_names,
        "roi_descriptions": roi_descriptions,
        "requested_seconds": float(seconds),
        "sleep_hint_seconds": float(sleep_hint),
        "buffer_seconds": float(_FIRST_SLEEP_BUFFER_SECONDS),
        "sleep_seconds": float(sleep_s),
        "max_total_seconds": float(max_total_seconds),
        "wait_logic": dict(wait_logic),
        "images": {k: f"before_{k}.png" for k in roi_names},
        "llm": dict(choice),
    }
    if llm_error:
        meta["llm_error"] = llm_error
    step.write_meta(meta)

    if sleep_s > 0:
        agent.logger.narrate(
            f"[WaitUntil] Sleeping once for {sleep_s:.1f}s (hint={sleep_hint:.1f}s + buffer={_FIRST_SLEEP_BUFFER_SECONDS:.0f}s)."
        )
        time.sleep(float(sleep_s))

    result: dict[str, Any] = {
        "rois": roi_names,
        "final_seconds": float(sleep_s),
        "rounds_used": 1,
        "total_waited_seconds": float(sleep_s),
        "log_root": str(agent.logger.run_root),
        "last_step": last_step_rel,
        "last_image": last_image_rel,
        "last_images": last_images,
        "llm_reason": str(choice.get("reason", "")).strip(),
        "sleep_hint_seconds": float(sleep_hint),
        "buffer_seconds": float(_FIRST_SLEEP_BUFFER_SECONDS),
    }

    agent._last_action_log = f"wait_until(rois={roi_names}, slept={float(sleep_s):.1f})"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "wait_until", "action_input": dict(action_input), "result": result, "say": say})
    agent._emit("result", step=step_index, action="wait_until", result=result)
    return "continue"
