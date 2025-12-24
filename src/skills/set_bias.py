from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

from ..actions import Actor
from ..capture import ScreenCapturer
from ..logger import RunLogger
from ..workspace import Workspace


@dataclass(frozen=True, slots=True)
class SetBiasParams:
    # Target is for logging/safety; entry text is what actually gets typed.
    target_value: float
    target_unit: Literal["mV", "V"] = "mV"
    typed_text: Optional[str] = None


def _convert(value: float, from_unit: Literal["mV", "V"], to_unit: Literal["mV", "V"]) -> float:
    if from_unit == to_unit:
        return value
    if from_unit == "V" and to_unit == "mV":
        return value * 1000.0
    if from_unit == "mV" and to_unit == "V":
        return value / 1000.0
    return value


def _format_number(value: float, *, decimals: int) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _infer_input_unit(workspace: Workspace) -> Literal["mV", "V"]:
    tool_cfg = (workspace.tools or {}).get("SetBias", {})
    if isinstance(tool_cfg, dict):
        cfg = str(tool_cfg.get("input_unit", "")).strip().lower()
        if cfg in {"v", "volt", "volts"}:
            return "V"
        if cfg in {"mv", "millivolt", "millivolts"}:
            return "mV"
    return "mV"


def run(
    *,
    workspace: Workspace,
    capturer: ScreenCapturer,
    actor: Actor,
    logger: RunLogger,
    params: SetBiasParams,
) -> Mapping[str, Any]:
    roi = workspace.roi("bias_readout")
    anchor = workspace.anchor("bias_input")

    tool_cfg = (workspace.tools or {}).get("SetBias", {})
    safety = tool_cfg.get("safety", {}) if isinstance(tool_cfg, dict) else {}
    min_mv = safety.get("min_mV", None)
    max_mv = safety.get("max_mV", None)

    input_unit = _infer_input_unit(workspace)
    if params.typed_text is not None and params.typed_text.strip():
        typed_text = params.typed_text.strip()
    else:
        target_in_input_unit = _convert(params.target_value, params.target_unit, input_unit)
        decimals = 6 if input_unit == "V" else 3
        typed_text = _format_number(target_in_input_unit, decimals=decimals)

    target_mv = _convert(params.target_value, params.target_unit, "mV")
    typed_mv: Optional[float] = None
    try:
        typed_float = float(typed_text)
        typed_mv = _convert(typed_float, input_unit, "mV")
    except Exception:
        typed_mv = None

    # Safety limits must apply to what we will actually type (typed_text), since that's what changes the UI.
    safety_mv = typed_mv if typed_mv is not None else target_mv
    if isinstance(min_mv, (int, float)) and safety_mv < float(min_mv):
        raise ValueError(f"Refusing SetBias({safety_mv:g} mV): below min_mV={min_mv}")
    if isinstance(max_mv, (int, float)) and safety_mv > float(max_mv):
        raise ValueError(f"Refusing SetBias({safety_mv:g} mV): above max_mV={max_mv}")

    step = logger.start_step("set_bias")
    logger.narrate(f"[SetBias] Setting bias to {params.target_value:g} {params.target_unit} (typing {typed_text!r})")

    before_img = capturer.capture_roi(roi)
    step.save_image("before.png", before_img)
    step.save_image("before_bias_readout.png", before_img)

    # Nanonis numeric inputs can be finicky: ensure focus, then aggressively clear before typing.
    actor.double_click(anchor)
    actor.hotkey("ctrl", "a")
    actor.press("backspace")
    actor.press("backspace")

    actor.click(anchor)
    actor.press("home")
    actor.hotkey("shift", "end")
    actor.press("backspace")

    actor.type_text(typed_text)
    actor.press("enter")

    after_img = capturer.capture_roi(roi)
    step.save_image("after.png", after_img)
    step.save_image("after_bias_readout.png", after_img)

    meta: dict[str, Any] = {
        "skill": "SetBias",
        "target_value": params.target_value,
        "target_unit": params.target_unit,
        "target_mV": target_mv,
        "input_unit": input_unit,
        "typed_text": typed_text,
        "typed_mV": typed_mv,
        "verified": None,
        "verification": "agent_vlm",  # verification is done by the multimodal agent using ROI images
    }
    step.write_meta(meta)
    return meta
