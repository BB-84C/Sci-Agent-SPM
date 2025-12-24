from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from PIL import Image


def _safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    return name or "step"


@dataclass(frozen=True, slots=True)
class StepLogger:
    path: Path

    def save_image(self, filename: str, img: Image.Image) -> None:
        out = self.path / filename
        img.save(out, format="PNG")

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


class RunLogger:
    def __init__(self, root_dir: str | Path = "logs", *, narrator: Optional[Callable[[str], None]] = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_root = Path(root_dir) / ts
        self.run_root.mkdir(parents=True, exist_ok=False)
        self._narrator = narrator

    def start_step(self, step_name: str) -> StepLogger:
        base = self.run_root / _safe_name(step_name)
        p = base
        i = 2
        while p.exists():
            p = self.run_root / f"{base.name}_{i}"
            i += 1
        p.mkdir(parents=True, exist_ok=False)
        return StepLogger(path=p)

    def narrate(self, text: str) -> None:
        if self._narrator is not None:
            try:
                self._narrator(text)
                return
            except Exception:
                pass
        print(text, flush=True)
