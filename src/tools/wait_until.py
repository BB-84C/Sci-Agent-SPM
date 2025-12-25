from __future__ import annotations

import time
from typing import Any, Mapping, Optional, TYPE_CHECKING, Literal

from ..llm_client import _extract_json_object, _img_to_data_url

if TYPE_CHECKING:
    from ..agent import VisualAutomationAgent


_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MAX_TOTAL_SECONDS = 6 * 60 * 60
_SLEEP_CAP_SECONDS = 6 * 60 * 60
_DEFAULT_NEXT_SECONDS = 10


def _require_roi_name(action_input: Mapping[str, Any]) -> str:
    roi = action_input.get("roi", None)
    if not isinstance(roi, str) or not roi.strip():
        raise ValueError("wait_until requires action_input.roi (ROI name)")
    return roi.strip()


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


def _coerce_suggest_seconds(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except Exception:
            return None
    return None


def _choose_next_seconds(*, suggested: Optional[int]) -> int:
    if suggested is None:
        return _DEFAULT_NEXT_SECONDS
    if suggested <= 0:
        return _DEFAULT_NEXT_SECONDS
    return int(suggested)


def _llm_confirm_done(
    agent: "VisualAutomationAgent",
    *,
    roi_name: str,
    roi_description: str,
    wait_reason: str,
    img: Any,
) -> Mapping[str, Any]:
    system = (
        "You decide whether to stop waiting by inspecting a UI ROI screenshot.\n"
        "Return ONLY one JSON object with exactly these keys:\n"
        '{ "done": true|false, "reason": "<short>", "suggest_seconds": <int optional> }\n'
        "No markdown. No code fences. No extra text."
    )
    user = (
        "Is the awaited condition satisfied (i.e., should we stop waiting)?\n\n"
        f"ROI name: {roi_name}\n"
        f"ROI description: {roi_description or '(none)'}\n"
        f"Wait reason: {wait_reason or '(none)'}\n"
    )

    resp = agent.llm._client.responses.create(  # type: ignore[attr-defined]
        model=agent.llm._config.model,  # type: ignore[attr-defined]
        input=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user},
                    {"type": "input_image", "image_url": _img_to_data_url(img)},
                ],
            },
        ],
        timeout=agent.llm._config.timeout_s,  # type: ignore[attr-defined]
    )
    agent.llm._set_last_usage(resp)  # type: ignore[attr-defined]
    agent._accumulate_last_usage()

    text = getattr(resp, "output_text", None) or ""
    obj = _extract_json_object(text)
    done = _coerce_done(obj.get("done", None))
    out: dict[str, Any] = {"done": done, "reason": str(obj.get("reason", "")).strip()}
    suggest = _coerce_suggest_seconds(obj.get("suggest_seconds", None))
    if suggest is not None:
        out["suggest_seconds"] = int(suggest)
    return out


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

    roi_name = _require_roi_name(action_input)
    seconds = _require_nonneg_seconds(action_input, "seconds")
    max_rounds = _optional_pos_int(action_input, "max_rounds", _DEFAULT_MAX_ROUNDS)
    max_total_seconds = _require_nonneg_seconds(
        {"max_total_seconds": action_input.get("max_total_seconds", _DEFAULT_MAX_TOTAL_SECONDS)},
        "max_total_seconds",
    )
    wait_reason = str(action_input.get("reason", "")).strip()

    try:
        roi = agent.workspace.roi(roi_name)
    except KeyError as e:
        step = agent.logger.start_step(f"wait_until_{roi_name}_invalid")
        meta = {"tool": "wait_until", "reason": "invalid_roi", "roi": roi_name, "error": str(e)}
        step.write_meta(meta)
        result = {"reason": "wait_until_invalid_roi", "roi": roi_name, "log_root": str(agent.logger.run_root)}
        results.append({"action": "wait_until", "action_input": dict(action_input), "result": result, "say": say})
        agent._emit("result", step=step_index, action="wait_until", result=result)
        return "continue"

    roi_description = getattr(roi, "description", "") or ""

    total_waited = 0.0
    rounds_used = 0
    last_image_rel = ""
    last_step_rel = ""
    last_sleep_s = 0.0

    agent.logger.narrate(f"[WaitUntil] Waiting on ROI {roi_name!r}.")

    for r in range(1, max_rounds + 1):
        rounds_used = r
        remaining = float(max_total_seconds) - total_waited
        if remaining <= 0:
            break

        sleep_s = min(float(seconds), remaining, float(_SLEEP_CAP_SECONDS))
        last_sleep_s = float(sleep_s)
        step = agent.logger.start_step(f"wait_until_{roi_name}_r{r}")
        last_step_rel = str(step.path.relative_to(agent.logger.run_root))

        if sleep_s > 0:
            agent.logger.narrate(f"[WaitUntil] Round {r}/{max_rounds}: waiting {sleep_s:.1f}s…")
            time.sleep(sleep_s)
            total_waited += sleep_s
        else:
            agent.logger.narrate(f"[WaitUntil] Round {r}/{max_rounds}: checking now…")

        img = agent.capturer.capture_roi(roi)
        img_name = f"after_{roi_name}.png"
        step.save_image(img_name, img)
        last_image_rel = str((step.path / img_name).relative_to(agent.logger.run_root))

        llm_error = ""
        confirm: Mapping[str, Any] = {}
        try:
            confirm = _llm_confirm_done(
                agent,
                roi_name=roi_name,
                roi_description=roi_description,
                wait_reason=wait_reason,
                img=img,
            )
        except Exception as e:
            llm_error = str(e)

        meta: dict[str, Any] = {
            "tool": "wait_until",
            "roi": roi_name,
            "roi_description": roi_description,
            "round": r,
            "max_rounds": max_rounds,
            "requested_seconds": float(seconds),
            "slept_seconds": float(sleep_s),
            "total_waited_seconds": float(total_waited),
            "max_total_seconds": float(max_total_seconds),
            "reason": wait_reason,
            "image": img_name,
        }
        if llm_error:
            meta["llm_error"] = llm_error
            step.write_meta(meta)
            break

        meta["llm"] = dict(confirm)
        step.write_meta(meta)

        if bool(confirm.get("done", False)):
            result: dict[str, Any] = {
                "roi": roi_name,
                "final_seconds": float(last_sleep_s),
                "rounds_used": rounds_used,
                "total_waited_seconds": float(total_waited),
                "log_root": str(agent.logger.run_root),
                "last_step": last_step_rel,
                "last_image": last_image_rel,
                "llm_reason": str(confirm.get("reason", "")).strip(),
            }
            agent._last_action_log = f"wait_until(roi={roi_name}, seconds={float(last_sleep_s):.1f})"
            agent._last_action_signature = signature
            agent._observed_since_last_action = False
            results.append({"action": "wait_until", "action_input": dict(action_input), "result": result, "say": say})
            agent._emit("result", step=step_index, action="wait_until", result=result)
            return "continue"

        suggested = _coerce_suggest_seconds(confirm.get("suggest_seconds", None))
        seconds = float(_choose_next_seconds(suggested=suggested))

    fail_reason = "wait_until_timeout"
    llm_error_val = locals().get("llm_error", "")
    if isinstance(llm_error_val, str) and llm_error_val:
        fail_reason = "wait_until_llm_error"

    fail_result: dict[str, Any] = {
        "reason": fail_reason,
        "roi": roi_name,
        "rounds_used": rounds_used,
        "total_waited_seconds": float(total_waited),
        "log_root": str(agent.logger.run_root),
        "last_step": last_step_rel,
        "last_image": last_image_rel,
    }
    if fail_reason == "wait_until_llm_error":
        fail_result["llm_error"] = str(llm_error_val)

    agent._last_action_log = f"wait_until(roi={roi_name}, failed=true)"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "wait_until", "action_input": dict(action_input), "result": fail_result, "say": say})
    agent._emit("result", step=step_index, action="wait_until", result=fail_result)
    return "continue"
