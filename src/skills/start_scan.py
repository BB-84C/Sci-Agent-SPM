from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..actions import Actor
from ..capture import ScreenCapturer
from ..logger import RunLogger
from ..workspace import Workspace


@dataclass(frozen=True, slots=True)
class StartScanParams:
    direction: Literal["top", "bottom", "default"] = "default"


def run(
    *,
    workspace: Workspace,
    capturer: ScreenCapturer,
    actor: Actor,
    logger: RunLogger,
    params: StartScanParams,
) -> Mapping[str, Any]:
    roi = workspace.roi("scan_status")

    if params.direction == "top":
        candidate_anchors = ["scan_start_from_top_button", "scan_start_button"]
        label = "Start Scan (from top)"
    elif params.direction == "bottom":
        candidate_anchors = ["scan_start_from_bottom_button", "scan_start_button"]
        label = "Start Scan (from bottom)"
    else:
        candidate_anchors = ["scan_start_button", "scan_start_from_top_button", "scan_start_from_bottom_button"]
        label = "Start Scan"

    anchor = None
    last_err: Exception | None = None
    for name in candidate_anchors:
        try:
            anchor = workspace.anchor(name)
            break
        except Exception as e:
            last_err = e
    if anchor is None:
        raise KeyError(
            "No suitable StartScan anchor found. Tried: "
            + ", ".join(repr(x) for x in candidate_anchors)
            + (f". Last error: {last_err}" if last_err else "")
        )

    step = logger.start_step("start_scan")
    logger.narrate(f"[StartScan] Clicking {label}...")

    before_img = capturer.capture_roi(roi)
    step.save_image("before.png", before_img)
    step.save_image("before_scan_status.png", before_img)

    actor.click(anchor)

    after_img = capturer.capture_roi(roi)
    step.save_image("after.png", after_img)
    step.save_image("after_scan_status.png", after_img)

    meta: dict[str, Any] = {
        "skill": "StartScan",
        "direction": params.direction,
        "anchor_used": anchor.name,
        "verified": False,
    }
    step.write_meta(meta)
    logger.narrate("[StartScan] Started (unverified); saved before/after screenshots.")
    return meta

