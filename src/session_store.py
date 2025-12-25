from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


SESSIONS_DIR = Path("sessions")


def _session_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name.strip())
    if not safe:
        safe = "session"
    return SESSIONS_DIR / f"{safe}.json"


def save_session(name: str, data: Mapping[str, Any]) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = _session_path(name)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def load_session(name: str) -> Optional[dict[str, Any]]:
    p = _session_path(name)
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else None


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    out: list[str] = []
    for p in sorted(SESSIONS_DIR.glob("*.json")):
        stem = p.stem
        # Hide temporary / internal sessions from the UI.
        if stem.startswith(".") or stem.startswith("__temp"):
            continue
        out.append(stem)
    return out
