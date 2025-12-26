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
    roi_names: list[str],
    roi_descriptions: list[str],
    wait_reason: str,
    images: list[tuple[str, Any]],
) -> Mapping[str, Any]:
    system = (
        "You decide whether to stop waiting by inspecting a UI ROI screenshot.\n"
        "Return ONLY one JSON object with exactly these keys:\n"
        '{ "done": true|false, "reason": "<short>", "suggest_seconds": <int optional> }\n'
        "No markdown. No code fences. No extra text."
    )
    roi_lines = []
    for n, d in zip(roi_names, roi_descriptions):
        roi_lines.append(f"- {n}: {d or '(none)'}")
    roi_block = "\n".join(roi_lines) if roi_lines else "(none)"
    user = (
        "Is the awaited condition satisfied (i.e., should we stop waiting)?\n\n"
        f"ROIs:\n{roi_block}\n"
        f"Wait reason: {wait_reason or '(none)'}\n"
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

    roi_names = _coerce_roi_names(action_input)
    seconds = _require_nonneg_seconds(action_input, "seconds")
    max_rounds = _optional_pos_int(action_input, "max_rounds", _DEFAULT_MAX_ROUNDS)
    max_total_seconds = _require_nonneg_seconds(
        {"max_total_seconds": action_input.get("max_total_seconds", _DEFAULT_MAX_TOTAL_SECONDS)},
        "max_total_seconds",
    )
    wait_reason = str(action_input.get("reason", "")).strip()

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

    total_waited = 0.0
    rounds_used = 0
    last_image_rel = ""
    last_step_rel = ""
    last_sleep_s = 0.0

    agent.logger.narrate(f"[WaitUntil] Waiting on ROI(s): {', '.join(roi_names)}.")

    for r in range(1, max_rounds + 1):
        rounds_used = r
        remaining = float(max_total_seconds) - total_waited
        if remaining <= 0:
            break

        sleep_s = min(float(seconds), remaining, float(_SLEEP_CAP_SECONDS))
        last_sleep_s = float(sleep_s)
        roi_label = roi_names[0] if len(roi_names) == 1 else f"{roi_names[0]}+{len(roi_names) - 1}"
        step = agent.logger.start_step(f"wait_until_{roi_label}_r{r}")
        last_step_rel = str(step.path.relative_to(agent.logger.run_root))

        if sleep_s > 0:
            agent.logger.narrate(f"[WaitUntil] Round {r}/{max_rounds}: waiting {sleep_s:.1f}s…")
            time.sleep(sleep_s)
            total_waited += sleep_s
        else:
            agent.logger.narrate(f"[WaitUntil] Round {r}/{max_rounds}: checking now…")

        captured: list[tuple[str, Any]] = []
        last_images: dict[str, str] = {}
        for roi_name, roi in zip(roi_names, resolved):
            img = agent.capturer.capture_roi(roi)
            img_name = f"after_{roi_name}.png"
            step.save_image(img_name, img)
            rel = str((step.path / img_name).relative_to(agent.logger.run_root))
            last_images[roi_name] = rel
            captured.append((roi_name, img))
        # Keep compatibility with older consumers expecting a single "last_image".
        last_image_rel = last_images.get(roi_names[0], "")

        llm_error = ""
        confirm: Mapping[str, Any] = {}
        try:
            confirm = _llm_confirm_done(
                agent,
                roi_names=roi_names,
                roi_descriptions=roi_descriptions,
                wait_reason=wait_reason,
                images=captured,
            )
        except Exception as e:
            llm_error = str(e)

        meta: dict[str, Any] = {
            "tool": "wait_until",
            "rois": roi_names,
            "roi_descriptions": roi_descriptions,
            "round": r,
            "max_rounds": max_rounds,
            "requested_seconds": float(seconds),
            "slept_seconds": float(sleep_s),
            "total_waited_seconds": float(total_waited),
            "max_total_seconds": float(max_total_seconds),
            "reason": wait_reason,
            "images": {k: f"after_{k}.png" for k in roi_names},
        }
        if llm_error:
            meta["llm_error"] = llm_error
            step.write_meta(meta)
            break

        meta["llm"] = dict(confirm)
        step.write_meta(meta)

        if bool(confirm.get("done", False)):
            result: dict[str, Any] = {
                "rois": roi_names,
                "final_seconds": float(last_sleep_s),
                "rounds_used": rounds_used,
                "total_waited_seconds": float(total_waited),
                "log_root": str(agent.logger.run_root),
                "last_step": last_step_rel,
                "last_image": last_image_rel,
                "last_images": last_images,
                "llm_reason": str(confirm.get("reason", "")).strip(),
            }
            agent._last_action_log = f"wait_until(rois={roi_names}, seconds={float(last_sleep_s):.1f})"
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
        "rois": roi_names,
        "rounds_used": rounds_used,
        "total_waited_seconds": float(total_waited),
        "log_root": str(agent.logger.run_root),
        "last_step": last_step_rel,
        "last_image": last_image_rel,
    }
    if fail_reason == "wait_until_llm_error":
        fail_result["llm_error"] = str(llm_error_val)

    agent._last_action_log = f"wait_until(rois={roi_names}, failed=true)"
    agent._last_action_signature = signature
    agent._observed_since_last_action = False
    results.append({"action": "wait_until", "action_input": dict(action_input), "result": fail_result, "say": say})
    agent._emit("result", step=step_index, action="wait_until", result=fail_result)
    return "continue"
