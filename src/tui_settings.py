from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_SETTINGS_PATH = Path("sessions") / ".tui_settings.json"


@dataclass(frozen=True, slots=True)
class TuiSettings:
    workspace: Optional[str] = None
    agent_model: Optional[str] = None
    tool_call_model: Optional[str] = None
    max_agent_steps: Optional[int] = None
    action_delay_s: Optional[float] = None
    abort_hotkey: Optional[bool] = None
    log_dir: Optional[str] = None
    memory_turns: Optional[int] = None
    mode: Optional[str] = None  # agent|chat|auto
    memory_compress_threshold_tokens: Optional[int] = None


def _coerce_str(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s if s else None


def _coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v.strip()))
        except Exception:
            return None
    return None


def _coerce_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return None


def load_tui_settings(path: Path = DEFAULT_SETTINGS_PATH) -> TuiSettings:
    try:
        if not path.exists():
            return TuiSettings()
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return TuiSettings()
    except Exception:
        return TuiSettings()

    return TuiSettings(
        workspace=_coerce_str(obj.get("workspace")),
        agent_model=_coerce_str(obj.get("agent_model")),
        tool_call_model=_coerce_str(obj.get("tool_call_model")),
        max_agent_steps=_coerce_int(obj.get("max_agent_steps")),
        action_delay_s=_coerce_float(obj.get("action_delay_s")),
        abort_hotkey=_coerce_bool(obj.get("abort_hotkey")),
        log_dir=_coerce_str(obj.get("log_dir")),
        memory_turns=_coerce_int(obj.get("memory_turns")),
        mode=_coerce_str(obj.get("mode")),
        memory_compress_threshold_tokens=_coerce_int(obj.get("memory_compress_threshold_tokens")),
    )


def save_tui_settings(data: TuiSettings, path: Path = DEFAULT_SETTINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for k, v in asdict(data).items():
        if v is None:
            continue
        payload[k] = v
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def merge_settings(base: TuiSettings, overrides: Mapping[str, Any]) -> TuiSettings:
    """
    Merge overrides into base, dropping unknown keys.
    """
    o = dict(overrides or {})
    return TuiSettings(
        workspace=_coerce_str(o.get("workspace", base.workspace)),
        agent_model=_coerce_str(o.get("agent_model", base.agent_model)),
        tool_call_model=_coerce_str(o.get("tool_call_model", base.tool_call_model)),
        max_agent_steps=_coerce_int(o.get("max_agent_steps", base.max_agent_steps)),
        action_delay_s=_coerce_float(o.get("action_delay_s", base.action_delay_s)),
        abort_hotkey=_coerce_bool(o.get("abort_hotkey", base.abort_hotkey)),
        log_dir=_coerce_str(o.get("log_dir", base.log_dir)),
        memory_turns=_coerce_int(o.get("memory_turns", base.memory_turns)),
        mode=_coerce_str(o.get("mode", base.mode)),
        memory_compress_threshold_tokens=_coerce_int(
            o.get("memory_compress_threshold_tokens", base.memory_compress_threshold_tokens)
        ),
    )
