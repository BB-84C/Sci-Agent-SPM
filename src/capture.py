from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mss
from PIL import Image

from .workspace import Roi


@dataclass(frozen=True, slots=True)
class ScreenCapturer:
    monitor_index: Optional[int] = None  # None => all monitors merged (mss monitor 0)

    def capture_fullscreen(self) -> Image.Image:
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index or 0]
            shot = sct.grab(monitor)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def capture_roi(self, roi: Roi) -> Image.Image:
        with mss.mss() as sct:
            shot = sct.grab({"left": roi.x, "top": roi.y, "width": roi.w, "height": roi.h})
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

