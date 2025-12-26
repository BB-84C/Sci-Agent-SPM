from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Roi:
    name: str
    x: int
    y: int
    w: int
    h: int
    description: str = ""
    tags: tuple[str, ...] = ()

    def as_bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True, slots=True)
class Anchor:
    name: str
    x: int
    y: int
    description: str = ""
    tags: tuple[str, ...] = ()
    linked_rois: tuple[str, ...] = ()

    def as_point(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Workspace:
    rois: tuple[Roi, ...]
    anchors: tuple[Anchor, ...]
    tools: Mapping[str, Any]
    source_path: Path

    def roi(self, name: str) -> Roi:
        for r in self.rois:
            if r.name == name:
                return r
        raise KeyError(f"ROI not found: {name!r}. Available: {', '.join(sorted(r.name for r in self.rois))}")

    def anchor(self, name: str) -> Anchor:
        for a in self.anchors:
            if a.name == name:
                return a
        raise KeyError(
            f"Anchor not found: {name!r}. Available: {', '.join(sorted(a.name for a in self.anchors))}"
        )


def _parse_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(x) for x in value)
    return (str(value),)


def _require_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key, None)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {key!r}, got: {value!r}")
    return value


def load_workspace(path: str | Path) -> Workspace:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workspace.json must be a JSON object")

    rois_raw = raw.get("rois", [])
    anchors_raw = raw.get("anchors", [])
    tools_raw = raw.get("tools", {}) or {}

    if not isinstance(rois_raw, list):
        raise ValueError("'rois' must be a list")
    if not isinstance(anchors_raw, list):
        raise ValueError("'anchors' must be a list")
    if not isinstance(tools_raw, dict):
        raise ValueError("'tools' must be a JSON object mapping tool names to config")

    rois: list[Roi] = []
    for item in rois_raw:
        if not isinstance(item, dict):
            raise ValueError("Each ROI must be an object")
        rois.append(
            Roi(
                name=str(item.get("name", "")),
                x=_require_int(item, "x"),
                y=_require_int(item, "y"),
                w=_require_int(item, "w"),
                h=_require_int(item, "h"),
                description=str(item.get("description", "")),
                tags=_parse_tags(item.get("tags")),
            )
        )

    anchors: list[Anchor] = []
    roi_names = {r.name for r in rois}
    for item in anchors_raw:
        if not isinstance(item, dict):
            raise ValueError("Each anchor must be an object")
        linked_raw = item.get("linked_ROIs", None)
        linked: tuple[str, ...] = ()
        if isinstance(linked_raw, list):
            linked = tuple(str(x) for x in linked_raw if isinstance(x, (str, int, float)) and str(x).strip())
        elif linked_raw is not None:
            linked = (str(linked_raw).strip(),) if str(linked_raw).strip() else ()
        # Keep only ROIs that exist (calibrator enforces this, but workspace files may drift).
        linked = tuple(x for x in linked if x in roi_names)
        anchors.append(
            Anchor(
                name=str(item.get("name", "")),
                x=_require_int(item, "x"),
                y=_require_int(item, "y"),
                description=str(item.get("description", "")),
                tags=_parse_tags(item.get("tags")),
                linked_rois=linked,
            )
        )

    for r in rois:
        if not r.name:
            raise ValueError("ROI 'name' is required")
        if r.w <= 0 or r.h <= 0:
            raise ValueError(f"ROI {r.name!r} must have positive w/h")
    for a in anchors:
        if not a.name:
            raise ValueError("Anchor 'name' is required")

    return Workspace(rois=tuple(rois), anchors=tuple(anchors), tools=tools_raw, source_path=p)
