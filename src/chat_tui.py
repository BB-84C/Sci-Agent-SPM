from __future__ import annotations

import ast
import json
import os
import re
import threading
import shutil
import subprocess
import sys
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, RichLog, Static, TextArea

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from .agent import VisualAutomationAgent
from .session_store import list_sessions, load_session, save_session
from .tui_settings import DEFAULT_SETTINGS_PATH, TuiSettings, load_tui_settings, save_tui_settings


@dataclass(frozen=True, slots=True)
class Theme:
    # Fallout Terminal palette
    bg: str = "#0C0C0C"
    fg: str = "#26E476"
    dim: str = "#1DB45E"
    user: str = "#61D6D6"
    agent: str = "#B4009E"
    accent: str = "#3B78FF"
    error: str = "#E74856"

    def mix(self, other: str, alpha: float) -> str:
        """Mix `other` into `bg` by alpha (0..1)."""

        def clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

        alpha = clamp01(alpha)

        def parse_hex(c: str) -> tuple[int, int, int]:
            c = c.lstrip("#")
            return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))

        def to_hex(rgb: tuple[int, int, int]) -> str:
            r, g, b = rgb
            return f"#{r:02X}{g:02X}{b:02X}"

        r0, g0, b0 = parse_hex(self.bg)
        r1, g1, b1 = parse_hex(other)
        r = round(r0 * (1.0 - alpha) + r1 * alpha)
        g = round(g0 * (1.0 - alpha) + g1 * alpha)
        b = round(b0 * (1.0 - alpha) + b1 * alpha)
        return to_hex((r, g, b))


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    tag: str
    content: str


def _parse_transcript(text: str) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("[") and "]" in line:
            close = line.find("]")
            tag = line[1:close].strip()
            rest = line[close + 1 :]
            if rest.startswith(" "):
                rest = rest[1:]
            content_lines: list[str] = []
            if rest:
                content_lines.append(rest)
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                # Continuation of a multi-line message / block.
                # Stop if a new [TAG] block begins without a blank separator.
                if lines[i].startswith("[") and "]" in lines[i]:
                    nxt_close = lines[i].find("]")
                    nxt_tag = lines[i][1:nxt_close].strip()
                    if nxt_tag and nxt_tag.replace(" ", "").isupper():
                        break
                content_lines.append(lines[i])
                i += 1
            entries.append(TranscriptEntry(tag=tag, content="\n".join(content_lines).rstrip()))
            continue

        entries.append(TranscriptEntry(tag="TEXT", content=line.rstrip()))
        i += 1
    return entries


class AgentEvent(Message):
    def __init__(self, ev: Mapping[str, Any]) -> None:
        super().__init__()
        self.ev = dict(ev)


class InputSubmitted(Message):
    bubble = True

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ChatApp(App[None]):
    CSS = """
    Screen {
        background: #0C0C0C;
    }
    #root {
        height: 100%;
        background: #0C0C0C;
    }
    #transcript {
        border: round #26E476;
        background: #0C0C0C;
        color: #26E476;
        padding: 1 2;
        scrollbar-color: #26E476;
    }
    #status {
        height: 1;
        color: #1DB45E;
        padding: 0 1;
    }
    #inputbar {
        background: #0C0C0C;
    }
    #inputbox {
        border: round #B4009E;
        background: #0C0C0C;
        padding: 0 1;
        width: 100%;
    }
    #prompt {
        width: 3;
        content-align: center middle;
        color: #B4009E;
        background: #0C0C0C;
    }
    #input {
        background: #0C0C0C;
        color: #26E476;
        border: none;
        padding: 0 0;
    }
    #input:focus {
        border: none;
    }
    #underbar {
        height: 1;
        background: #0C0C0C;
        padding: 0 1;
    }
    #underleft {
        width: 1fr;
    }
    #tokens {
        width: 60;
        content-align: right middle;
        color: #1DB45E;
        background: #0C0C0C;
    }
    #cache {
        width: 42;
        content-align: left middle;
        color: #1DB45E;
        background: #0C0C0C;
        padding: 0 1;
    }
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "focus_transcript", "Log"),
        ("ctrl+i", "focus_input", "Input"),
        ("ctrl+shift+c", "copy_mode", "Select/Copy"),
        ("ctrl+c", "abort", "Abort"),
        Binding("ctrl+s", "submit_input", "Send", show=True, priority=True),
        Binding("ctrl+enter", "submit_input", show=False, priority=True),
        Binding("ctrl+kp_enter", "submit_input", show=False, priority=True),
        ("pageup", "scroll_up", "Up"),
        ("pagedown", "scroll_down", "Down"),
    ]

    busy: bool = reactive(False)

    def __init__(self, *, agent: VisualAutomationAgent, first_message: Optional[str] = None) -> None:
        super().__init__()
        self._agent = agent
        self._first_message = first_message.strip() if first_message else None
        self._theme = Theme()
        self._sigint_event = threading.Event()
        self._history_display: list[str] = []
        self._history_full: list[str] = []
        self._history_entries: list[dict[str, Any]] = []
        self._events: list[Mapping[str, Any]] = []
        self._confirm_mode: Optional[str] = None
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._tokens_total: int = 0
        # Show full tool call/result payloads in the transcript (no truncation/collapse).
        self._truncate_tool_blocks: bool = False
        self._temp_session_path = Path("sessions") / ".temp_session.json"
        self._current_session_name: Optional[str] = None
        self._current_session_saved: bool = False
        self._session_log_roots: set[str] = set()
        self._cache_usage_inflight: bool = False
        self._cache_usage_text: str = ""
        self._settings_path = DEFAULT_SETTINGS_PATH

    def _persisted_settings_snapshot(self) -> TuiSettings:
        cfg = self._agent.config
        try:
            ws_path = str(self._agent.workspace.source_path)
        except Exception:
            ws_path = None
        return TuiSettings(
            workspace=ws_path,
            agent_model=str(getattr(cfg, "agent_model", "") or "").strip() or None,
            tool_call_model=str(getattr(cfg, "tool_call_model", "") or "").strip() or None,
            max_agent_steps=int(getattr(cfg, "max_steps", 0)) if int(getattr(cfg, "max_steps", 0)) > 0 else None,
            action_delay_s=float(getattr(cfg, "action_delay_s", 0.0))
            if float(getattr(cfg, "action_delay_s", 0.0)) >= 0
            else None,
            abort_hotkey=bool(getattr(cfg, "abort_hotkey", True)),
            log_dir=str(getattr(cfg, "log_dir", "") or "").strip() or None,
            memory_turns=int(getattr(cfg, "memory_turns", -1)),
            mode=str(getattr(cfg, "run_mode", "") or "").strip() or None,
            memory_compress_threshold_tokens=int(getattr(cfg, "memory_compress_threshold_tokens", 300_000)),
        )

    def _save_persisted_settings(self) -> None:
        try:
            save_tui_settings(self._persisted_settings_snapshot(), self._settings_path)
        except Exception:
            pass

    def _load_persisted_settings(self) -> None:
        try:
            st = load_tui_settings(self._settings_path)
        except Exception:
            return
        try:
            if st.workspace:
                self._agent.set_workspace(st.workspace)
        except Exception:
            pass
        try:
            if st.agent_model:
                self._agent.set_agent_model(st.agent_model)
        except Exception:
            pass
        try:
            if st.tool_call_model:
                self._agent.set_tool_call_model(st.tool_call_model)
        except Exception:
            pass
        try:
            if st.max_agent_steps is not None:
                self._agent.set_max_steps(int(st.max_agent_steps))
        except Exception:
            pass
        try:
            if st.action_delay_s is not None:
                self._agent.set_action_delay_s(float(st.action_delay_s))
        except Exception:
            pass
        try:
            if st.abort_hotkey is not None:
                self._agent.set_abort_hotkey(bool(st.abort_hotkey))
        except Exception:
            pass
        try:
            if st.log_dir:
                self._agent.set_log_dir(st.log_dir)
        except Exception:
            pass
        try:
            if st.memory_turns is not None:
                self._agent.set_memory_turns(int(st.memory_turns))
        except Exception:
            pass
        try:
            if st.mode:
                self._agent.set_run_mode(st.mode)
        except Exception:
            pass
        try:
            if st.memory_compress_threshold_tokens is not None:
                self._agent.set_memory_compress_threshold_tokens(int(st.memory_compress_threshold_tokens))
        except Exception:
            pass

    def on_unmount(self) -> None:
        if not self._current_session_saved:
            self._delete_log_roots(self._session_log_roots)
        # Keep `.temp_session.json` on disk for crash recovery / post-run inspection.
        # The temp session is cleared explicitly when the user starts a new session.
        self._save_persisted_settings()

    def _clear_temp_session(self) -> None:
        try:
            if self._temp_session_path.exists():
                self._temp_session_path.unlink()
        except Exception:
            pass

    def _current_log_root(self) -> str:
        try:
            return str(self._agent.logger.run_root)
        except Exception:
            return ""

    def _track_log_root(self, root: str) -> None:
        root = str(root or "").strip()
        if not root:
            return
        self._session_log_roots.add(root)

    def _delete_log_roots(self, roots: Any) -> None:
        try:
            iterable = list(roots)
        except Exception:
            iterable = []
        for r in set(str(x).strip() for x in iterable if str(x).strip()):
            try:
                p = Path(r)
                if p.exists() and p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                continue

    def _reset_run_logger(self) -> None:
        # Create a fresh log root for the next chat session.
        try:
            self._agent.set_log_dir(self._agent.config.log_dir)
        except Exception:
            pass
        self._session_log_roots = set()
        self._track_log_root(self._current_log_root())

    def _format_bytes(self, n: int) -> str:
        n = max(0, int(n))
        units = ["B", "KB", "MB", "GB", "TB"]
        x = float(n)
        u = 0
        while x >= 1024.0 and u < len(units) - 1:
            x /= 1024.0
            u += 1
        if u == 0:
            return f"{int(x)} {units[u]}"
        return f"{x:.1f} {units[u]}"

    def _dir_size_bytes(self, path: Path) -> int:
        try:
            if not path.exists() or not path.is_dir():
                return 0
        except Exception:
            return 0
        total = 0
        try:
            for root, _dirs, files in os.walk(str(path)):
                for fn in files:
                    try:
                        fp = Path(root) / fn
                        total += fp.stat().st_size
                    except Exception:
                        continue
        except Exception:
            return 0
        return int(total)

    def _cache_roots(self) -> set[Path]:
        roots: set[Path] = {Path("logs")}
        try:
            roots.add(Path(self._agent.config.log_dir))
        except Exception:
            pass
        return roots

    def _request_cache_usage_update(self) -> None:
        if self._cache_usage_inflight:
            return
        self._cache_usage_inflight = True

        roots = list(self._cache_roots())
        session_roots = [Path(x) for x in self._session_log_roots if isinstance(x, str) and x.strip()]

        def compute() -> None:
            total_bytes = 0
            for r in roots:
                total_bytes += self._dir_size_bytes(r)
            session_bytes = 0
            for r in session_roots:
                session_bytes += self._dir_size_bytes(r)
            text = f"Cache: {self._format_bytes(total_bytes)}"
            if session_roots:
                text += f" (session: {self._format_bytes(session_bytes)})"

            def apply() -> None:
                self._cache_usage_inflight = False
                self._cache_usage_text = text
                try:
                    self.query_one("#cache", Static).update(text)
                except Exception:
                    pass

            try:
                self.call_from_thread(apply)
            except Exception:
                apply()

        threading.Thread(target=compute, daemon=True).start()

    def _format_ts_short(self, ts_iso: str) -> str:
        try:
            dt = datetime.fromisoformat(ts_iso)
            return dt.strftime("%H:%M:%S")
        except Exception:
            return ts_iso

    def _normalize_for_transcript(self, text: str) -> str:
        # The transcript parser treats *blank lines* as entry separators. If user/agent
        # content contains `\n\n`, the extra blank line would prematurely end the entry,
        # causing subsequent lines to lose the bubble/background styling.
        #
        # We keep the entry intact by replacing empty lines with a zero-width space
        # (not considered whitespace by `str.strip()`), which renders invisibly but
        # prevents the parser from terminating the block early.
        s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        parts = s.split("\n")
        if len(parts) <= 1:
            return s
        zws = "\u200B"
        parts = [p if p != "" else zws for p in parts]
        return "\n".join(parts)

    def _serialize_history_items(self, items: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tag = it.get("tag", None)
            content = it.get("content", None)
            ts = it.get("ts", None)
            if not isinstance(tag, str) or not isinstance(content, str):
                continue
            ts_short = self._format_ts_short(ts) if isinstance(ts, str) and ts.strip() else ""
            tag_u = tag.strip().upper()
            content_norm = self._normalize_for_transcript(content)
            if tag_u in {"YOU", "AGENT", "SYSTEM", "ERROR"}:
                prefix = f"[{tag_u}]"
                if ts_short:
                    prefix = f"{prefix} {ts_short}"
                chunks.append(f"{prefix} {content_norm}\n\n")
            else:
                header = f"[{tag_u}]"
                if ts_short:
                    header = f"{header} {ts_short}"
                chunks.append(f"{header}\n{content_norm}\n\n")
        return "".join(chunks)

    def _append_entry(self, *, tag: str, content: str, block: bool) -> None:
        ts_iso = datetime.now().isoformat(timespec="seconds")
        ts_short = self._format_ts_short(ts_iso)
        tag_u = tag.strip().upper()
        self._history_entries.append({"ts": ts_iso, "tag": tag_u, "content": content})
        content_norm = self._normalize_for_transcript(content)
        if tag_u in {"YOU", "AGENT", "SYSTEM", "ERROR"}:
            self._append_raw(f"[{tag_u}] {ts_short} {content_norm}\n\n")
            return
        if block:
            self._append_raw(f"[{tag_u}] {ts_short}\n{content_norm}\n\n")
        else:
            self._append_raw(f"[{tag_u}] {ts_short} {content_norm}\n\n")

    def _write_temp_session(self) -> None:
        try:
            self._temp_session_path.parent.mkdir(parents=True, exist_ok=True)
            history_text = "".join(self._history_full)
            agent_state = self._agent.export_session()
            entries = list(self._history_entries)
            state = {
                "name": "__temp__",
                "workspace": str(self._agent.workspace.source_path),
                # Prefer structured history. Keep the plain text for backwards compatibility.
                "history": entries,
                "history_text": history_text,
                "log_roots": sorted(self._session_log_roots),
                "agent": agent_state,
                "tokens": agent_state.get("tokens", {}),
            }
            self._temp_session_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    class ChatInput(TextArea):
        ALLOW_SELECT = True
        # Override TextArea defaults so selection/copy works like a normal editor.
        BINDINGS = [
            Binding("ctrl+a", "select_all", "Select all", show=True, priority=True),
        ]

    class Transcript(RichLog):
        pass

    class TranscriptCopyScreen(ModalScreen[None]):
        CSS = """
        TranscriptCopyScreen {
            background: rgba(0, 0, 0, 85%);
        }
        #copy_root {
            width: 90%;
            height: 85%;
            margin: 2 5;
            border: round #26E476;
            background: #0C0C0C;
        }
        #copy_area {
            width: 100%;
            height: 1fr;
            background: #0C0C0C;
            color: #26E476;
            border: none;
            padding: 1 2;
        }
        #copy_hint {
            height: 1;
            padding: 0 2;
            color: #1DB45E;
            background: #0C0C0C;
        }
        """

        BINDINGS = [
            Binding("escape", "dismiss", "Close", show=True, priority=True),
        ]

        def __init__(self, *, text: str) -> None:
            super().__init__()
            self._text = text

        def compose(self) -> ComposeResult:
            with Container(id="copy_root"):
                ta = TextArea(self._text, id="copy_area")
                ta.soft_wrap = True
                ta.show_line_numbers = False
                ta.show_horizontal_scrollbar = True
                ta.show_vertical_scrollbar = True
                ta.read_only = True
                yield ta
                yield Static("Select text with mouse/keys, Ctrl+C to copy, Esc to close.", id="copy_hint")

        def on_mount(self) -> None:
            self.query_one("#copy_area", TextArea).focus()

    def compose(self) -> ComposeResult:
        with Container(id="root"):
            transcript = self.Transcript(id="transcript", wrap=True, markup=False, highlight=False)
            yield transcript

            yield Static("", id="status")

            with Horizontal(id="inputbar"):
                with Horizontal(id="inputbox"):
                    yield Static(">", id="prompt")
                    inp = self.ChatInput("", id="input")
                    inp.soft_wrap = True
                    inp.show_line_numbers = False
                    inp.show_horizontal_scrollbar = False
                    inp.show_vertical_scrollbar = False
                    yield inp

            with Horizontal(id="underbar"):
                yield Static("", id="cache")
                yield Static("", id="underleft")
                yield Static("", id="tokens")

            yield Footer()

    def on_mount(self) -> None:
        self._load_persisted_settings()
        try:
            signal.signal(signal.SIGINT, lambda _sig, _frame: self._sigint_event.set())
        except Exception:
            pass
        try:
            self.set_interval(0.1, self._poll_sigint)
        except Exception:
            pass
        self._set_status("Ready")
        self._append_system(f"Workspace: {self._agent.workspace.source_path}")
        self._append_system("Use /help for help")
        if self._agent.config.abort_hotkey:
            self._append_system("Safety: Ctrl+C abort hotkey")
        else:
            self._append_system("Safety: Abort hotkey disabled, use /help to learn how to enable abort")
        self._sync_tokens_from_agent()
        self._track_log_root(self._current_log_root())
        self._request_cache_usage_update()
        inp = self.query_one("#input", TextArea)
        inp.focus()
        self._resize_input_to_content()
        if self._first_message:
            self.call_later(lambda: self._send(self._first_message or ""))

    def _poll_sigint(self) -> None:
        if not self._sigint_event.is_set():
            return
        self._sigint_event.clear()
        self.action_abort()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _set_tokens(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._tokens_in = max(0, int(input_tokens))
        self._tokens_out = max(0, int(output_tokens))
        self._tokens_total = max(0, int(total_tokens))
        self.query_one("#tokens", Static).update(
            f"Tokens: Send {self._tokens_in:,} / Received {self._tokens_out:,} / Total {self._tokens_total:,}"
        )

    def _sync_tokens_from_agent(self) -> None:
        try:
            st = self._agent.export_session()
            toks = st.get("tokens", {})
            if isinstance(toks, dict):
                self._set_tokens(
                    input_tokens=int(toks.get("input_tokens", 0)),
                    output_tokens=int(toks.get("output_tokens", 0)),
                    total_tokens=int(toks.get("total_tokens", 0)),
                )
                return
        except Exception:
            pass
        self._set_tokens(input_tokens=0, output_tokens=0, total_tokens=0)

    def _transcript(self) -> RichLog:
        return self.query_one("#transcript", RichLog)

    def _input(self) -> TextArea:
        return self.query_one("#input", TextArea)

    def _resize_input_to_content(self) -> None:
        inp = self._input()
        inputbox = self.query_one("#inputbox", Horizontal)
        inputbar = self.query_one("#inputbar", Horizontal)

        # Estimate wrapped lines using the *viewport* width, not the content width.
        # (content_size.width can grow with long lines and cause undercounting, which
        # then renders wrapped text outside the input box.)
        width = int(getattr(getattr(inp, "size", None), "width", 0))
        if width <= 0:
            width = max(20, int(getattr(getattr(self, "size", None), "width", 120)) - 12)

        def line_len(s: str) -> int:
            return len(s.replace("\t", "    "))

        total_lines = 0
        for raw in (inp.text or "").splitlines() or [""]:
            total_lines += max(1, (line_len(raw) + width - 1) // max(1, width))

        # Auto-expand up to 7 lines; keep scrollbars hidden.
        new_h = max(1, min(7, total_lines))

        inp.styles.height = new_h
        inp.styles.min_height = new_h
        inp.styles.max_height = new_h

        outer_h = new_h + 2
        inputbox.styles.height = outer_h
        inputbox.styles.min_height = outer_h
        inputbox.styles.max_height = outer_h
        inputbar.styles.height = outer_h
        inputbar.styles.min_height = outer_h
        inputbar.styles.max_height = outer_h
        self.refresh(layout=True)

    def _append_raw(self, text: str) -> None:
        self._history_full.append(text)
        self._history_display.append(text)
        self._render_transcript_chunk(text)
        self._write_temp_session()

    def _append_system(self, text: str) -> None:
        self._append_entry(tag="SYSTEM", content=text, block=False)

    def _append_user(self, text: str) -> None:
        self._append_entry(tag="YOU", content=text, block=False)

    def _append_agent(self, text: str) -> None:
        self._append_entry(tag="AGENT", content=text, block=False)

    def _append_block(self, title: str, content: str) -> None:
        self._append_entry(tag=title.upper(), content=content, block=True)

    def _append_error(self, text: str) -> None:
        self._append_entry(tag="ERROR", content=text, block=False)

    def _get_transcript_plain_text(self) -> str:
        return "".join(self._history_display)

    def _serialize_entries(self, entries: list[TranscriptEntry]) -> str:
        chunks: list[str] = []
        for e in entries:
            tag = e.tag.strip()
            content = e.content or ""
            if tag == "TEXT":
                chunks.append(content + "\n")
                continue
            if tag.upper() in {"YOU", "AGENT", "SYSTEM", "ERROR"}:
                chunks.append(f"[{tag.upper()}] {content}\n\n")
            else:
                chunks.append(f"[{tag.upper()}]\n{content}\n\n")
        return "".join(chunks)

    def _render_transcript_chunk(self, chunk: str) -> None:
        transcript = self._transcript()
        for entry in _parse_transcript(chunk):
            transcript.write(self._render_entry(entry))
            transcript.write(Text(""))

    def _display_entries(self, text: str) -> list[TranscriptEntry]:
        entries = _parse_transcript(text)
        if not self._truncate_tool_blocks:
            return entries

        last_response_idx = -1
        for idx, entry in enumerate(entries):
            tag = entry.tag.strip().upper()
            if tag in {"AGENT", "DONE"} and (entry.content or "").strip():
                last_response_idx = idx

        if last_response_idx < 0:
            return entries

        def parse_mapping(content: str) -> Optional[dict[str, Any]]:
            raw = (content or "").strip()
            if not raw:
                return None
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
            try:
                obj = ast.literal_eval(raw)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        def extract_tool_from_tool_call(content: str) -> Optional[str]:
            for line in (content or "").splitlines():
                m = re.match(r"\\s*tool\\s*:\\s*(.+?)\\s*$", line, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip() or None

            obj = parse_mapping(content)
            if isinstance(obj, dict):
                v = obj.get("tool", None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                v = obj.get("name", None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                fn = obj.get("function", None)
                if isinstance(fn, dict):
                    v = fn.get("name", None)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                for k in ("skill", "action"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()

            # Last-resort: try to extract `"tool": "..."` / `"name": "..."` patterns.
            m = re.search(r"\"(?:tool|name)\"\\s*:\\s*\"([^\"]+)\"", content or "")
            if m:
                return m.group(1).strip() or None
            return None

        def extract_args_from_tool_call(content: str) -> Optional[dict[str, Any]]:
            if not (content or "").strip():
                return None
            # Preferred format: `tool: ...\nargs:\n{...}`.
            lines = (content or "").splitlines()
            for i, line in enumerate(lines):
                # Single-line format: `args: {...}`
                m_inline = re.match(r"\\s*args\\s*:\\s*(\\{.*\\})\\s*$", line, flags=re.IGNORECASE)
                if m_inline:
                    obj = parse_mapping(m_inline.group(1))
                    return obj if isinstance(obj, dict) else None

                if re.match(r"\\s*args\\s*:\\s*$", line, flags=re.IGNORECASE):
                    raw = "\n".join(lines[i + 1 :]).strip()
                    obj = parse_mapping(raw)
                    return obj if isinstance(obj, dict) else None

            obj = parse_mapping(content)
            if not isinstance(obj, dict):
                return None
            # New format: { "tool": "...", "args": { ... } }
            args = obj.get("args", None)
            if isinstance(args, dict):
                return args
            # Alternate: { "name": "...", "arguments": {...} } or arguments as JSON string
            args = obj.get("arguments", None)
            if isinstance(args, dict):
                return args
            if isinstance(args, str) and args.strip():
                parsed = parse_mapping(args)
                if isinstance(parsed, dict):
                    return parsed
            fn = obj.get("function", None)
            if isinstance(fn, dict):
                args = fn.get("args", None)
                if isinstance(args, dict):
                    return args
                args = fn.get("arguments", None)
                if isinstance(args, dict):
                    return args
                if isinstance(args, str) and args.strip():
                    parsed = parse_mapping(args)
                    if isinstance(parsed, dict):
                        return parsed
            # Legacy fallback: the object itself is args
            return obj

        def extract_tool_or_skill_from_result(content: str) -> Optional[tuple[str, str]]:
            obj = parse_mapping(content)
            if not isinstance(obj, dict):
                return None
            v = obj.get("tool", None)
            if isinstance(v, str) and v.strip():
                return ("tool", v.strip())
            for k in ("skill", "tool", "name", "action"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    label = "skill" if k == "skill" else "tool"
                    return (label, v.strip())
            return None

        def tool_name_for_result_at(index: int) -> Optional[str]:
            # Prefer parsing from the result payload itself.
            payload = extract_tool_or_skill_from_result(entries[index].content or "")
            if payload is not None:
                _, name = payload
                return name

            # Fall back to the nearest previous tool call.
            for j in range(index - 1, -1, -1):
                if entries[j].tag.strip().upper() != "TOOL CALL":
                    continue
                return extract_tool_from_tool_call(entries[j].content or "")
            return None

        def reason_for_result(content: str) -> str:
            obj = parse_mapping(content)
            if not isinstance(obj, dict):
                return ""
            # New format: { "tool": "...", "result": { ... } }
            inner = obj.get("result", None)
            if isinstance(inner, dict):
                for k in ("reason", "llm_reason", "llm_error", "error"):
                    v = inner.get(k, None)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            for k in ("reason", "llm_reason", "llm_error", "error"):
                v = obj.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def nearest_tool_call_payload(index: int) -> tuple[Optional[str], Optional[dict[str, Any]]]:
            for j in range(index - 1, -1, -1):
                if entries[j].tag.strip().upper() != "TOOL CALL":
                    continue
                return (extract_tool_from_tool_call(entries[j].content or ""), extract_args_from_tool_call(entries[j].content or ""))
            return (None, None)

        out: list[TranscriptEntry] = []
        for idx, entry in enumerate(entries):
            tag = entry.tag.strip().upper()
            if idx < last_response_idx and tag in {"TOOL CALL", "TOOL RESULT"}:
                if tag == "TOOL CALL":
                    tool = extract_tool_from_tool_call(entry.content or "")
                    parsed_args = extract_args_from_tool_call(entry.content or "")
                    args = parsed_args if isinstance(parsed_args, dict) else {}
                    reason = ""
                    if isinstance(args, dict):
                        v = args.get("reason", None)
                        if isinstance(v, str) and v.strip():
                            reason = v.strip()
                    try:
                        args_json = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args_json = "{}"
                    if not tool:
                        fallback = (entry.content or "").strip().splitlines()[0:1]
                        tool = (fallback[0][:60] + "…") if fallback and len(fallback[0]) > 60 else (fallback[0] if fallback else "(unknown)")
                    collapsed_lines = [f"tool: {tool}"]
                    if reason:
                        collapsed_lines.append(f"reason: {reason}")
                    collapsed_lines.append(f"args: {args_json if parsed_args is not None else '(unparsed)'}")
                    collapsed = "\n".join(collapsed_lines)
                    out.append(TranscriptEntry(tag=entry.tag, content=collapsed))
                else:
                    payload = extract_tool_or_skill_from_result(entry.content or "")
                    tool = payload[1] if payload is not None else None
                    if tool is None:
                        tool, args = nearest_tool_call_payload(idx)
                    else:
                        _, args = nearest_tool_call_payload(idx)
                    reason = reason_for_result(entry.content or "")
                    try:
                        args_json = json.dumps(args or {}, ensure_ascii=False)
                    except Exception:
                        args_json = "{}"
                    if not tool:
                        tool = "(unknown)"
                    collapsed_lines = [f"tool: {tool}"]
                    if reason:
                        collapsed_lines.append(f"reason: {reason}")
                    collapsed_lines.append(f"args: {args_json}")
                    collapsed = "\n".join(collapsed_lines)
                    out.append(TranscriptEntry(tag=entry.tag, content=collapsed))
                continue
            out.append(entry)
        return out

    def _render_transcript_full(self, text: str) -> None:
        transcript = self._transcript()
        transcript.clear()
        for entry in self._display_entries(text):
            transcript.write(self._render_entry(entry))
            transcript.write(Text(""))

    def _render_entry(self, entry: TranscriptEntry) -> object:
        t = self._theme
        tag = entry.tag.strip().upper()
        content = entry.content or ""
        content_md = content.replace("\u200B", "")

        def looks_like_markdown(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            if "```" in s:
                return True
            if re.search(r"(?m)^\\s{0,3}[-*+]\\s+\\S", s):
                return True
            if re.search(r"(?m)^#{1,6}\\s+\\S", s):
                return True
            if re.search(r"\\*\\*[^*]+\\*\\*", s):
                return True
            if re.search(r"\\[[^\\]]+\\]\\([^)]+\\)", s):
                return True
            return False

        def split_ts_prefix(s: str) -> tuple[str, str]:
            raw = s or ""
            m = re.match(r"^(\\d\\d:\\d\\d:\\d\\d)([\\s\\n].*)$", raw)
            if not m:
                return ("", raw)
            ts = m.group(1)
            rest = (m.group(2) or "")
            if rest.startswith(" "):
                rest = rest[1:]
            if rest.startswith("\n"):
                rest = rest[1:]
            return (ts, rest)

        def append_multiline(out: Text, s: str, *, style: str) -> None:
            # In Rich, styles do not reliably carry across newline boundaries when
            # appending a single string containing '\n'. Append line-by-line instead.
            parts = s.splitlines(keepends=True)
            if not parts:
                return
            for part in parts:
                out.append(part, style=style)

        def chip(label: str, color: str) -> Text:
            return Text(f" {label} ", style=f"bold {t.bg} on {color}")

        def box_title(label: str, color: str) -> Text:
            return Text(f" {label} ", style=f"bold {t.bg} on {color}")

        if tag == "YOU":
            msg_bg = t.mix(t.user, 0.10)
            if looks_like_markdown(content_md):
                ts, body = split_ts_prefix(content_md)
                header = Text()
                header.append_text(chip("YOU", t.user))
                if ts:
                    header.append(f" {ts}", style=f"{t.user} on {msg_bg}")
                return Group(header, Markdown(body, style=f"{t.user} on {msg_bg}"), Text(""))
            msg = Text()
            msg.append_text(chip("YOU", t.user))
            msg.append(" ")
            append_multiline(msg, content, style=f"{t.user} on {msg_bg}")
            return msg

        if tag == "AGENT":
            msg_bg = t.mix(t.agent, 0.10)
            if looks_like_markdown(content_md):
                ts, body = split_ts_prefix(content_md)
                header = Text()
                header.append_text(chip("AGENT", t.agent))
                if ts:
                    header.append(f" {ts}", style=f"{t.fg} on {msg_bg}")
                return Group(header, Markdown(body, style=f"{t.fg} on {msg_bg}"), Text(""))
            msg = Text()
            msg.append_text(chip("AGENT", t.agent))
            msg.append(" ")
            append_multiline(msg, content, style=f"{t.fg} on {msg_bg}")
            return msg

        if tag == "SYSTEM":
            msg_bg = t.mix(t.dim, 0.08)
            if looks_like_markdown(content_md):
                ts, body = split_ts_prefix(content_md)
                header = Text()
                header.append_text(chip("SYSTEM", t.dim))
                if ts:
                    header.append(f" {ts}", style=f"{t.dim} on {msg_bg}")
                return Group(header, Markdown(body, style=f"{t.dim} on {msg_bg}"), Text(""))
            msg = Text()
            msg.append_text(chip("SYSTEM", t.dim))
            msg.append(" ")
            append_multiline(msg, content, style=f"{t.dim} on {msg_bg}")
            return msg

        if tag == "ERROR":
            msg_bg = t.mix(t.error, 0.08)
            if looks_like_markdown(content_md):
                ts, body = split_ts_prefix(content_md)
                header = Text()
                header.append_text(chip("ERROR", t.error))
                if ts:
                    header.append(f" {ts}", style=f"{t.error} on {msg_bg}")
                return Group(header, Markdown(body, style=f"{t.error} on {msg_bg}"), Text(""))
            msg = Text()
            msg.append_text(chip("ERROR", t.error))
            msg.append(" ")
            append_multiline(msg, content, style=f"{t.error} on {msg_bg}")
            return msg

        block_styles: dict[str, tuple[str, str]] = {
            "PLAN": ("PLAN", t.accent),
            "OBSERVATION": ("OBSERVATION", t.dim),
            "THOUGHT": ("THOUGHT", t.agent),
            "ACTION": ("ACTION", t.accent),
            "OBSERVE": ("OBSERVE", t.dim),
            "THINK": ("THINK", t.agent),
            "NEXT": ("NEXT", t.user),
            "TOOL CALL": ("TOOL CALL", t.accent),
            "TOOL RESULT": ("TOOL RESULT", t.user),
            "DONE": ("DONE", t.fg),
            "SESSIONS": ("SESSIONS", t.user),
            "HELP": ("HELP", t.dim),
            "CONFIRM": ("CONFIRM", t.accent),
            "WORKSPACE": ("WORKSPACE", t.dim),
            "MODEL": ("MODEL", t.dim),
            "AGENT MODEL": ("AGENT MODEL", t.dim),
            "TOOL CALL MODEL": ("TOOL CALL MODEL", t.dim),
            "MAX AGENT STEPS": ("MAX AGENT STEPS", t.dim),
            "ACTION DELAY": ("ACTION DELAY", t.dim),
            "ABORT HOTKEY": ("ABORT HOTKEY", t.dim),
            "LOG DIR": ("LOG DIR", t.dim),
        }
        if tag in block_styles:
            label, color = block_styles[tag]
            body_bg = t.mix(color, 0.08)
            if tag not in {"TOOL CALL", "TOOL RESULT"} and looks_like_markdown(content_md):
                ts, body = split_ts_prefix(content_md)
                header = Text()
                header.append_text(box_title(label, color))
                if ts:
                    header.append(f"\n{ts}", style=f"{t.fg} on {body_bg}")
                return Group(header, Markdown(body, style=f"{t.fg} on {body_bg}"), Text(""))
            out = Text()
            out.append_text(box_title(label, color))
            out.append("\n")
            append_multiline(out, content, style=f"{t.fg} on {body_bg}")
            return out

        # Fallback: show untagged lines without extra styling.
        if looks_like_markdown(content_md):
            return Markdown(content_md, style=t.fg)
        return Text(content, style=t.fg)

    def _rerender_transcript_from_history(self) -> None:
        # Render the full transcript in the UI (no truncation/collapse).
        self._history_display = list(self._history_full)
        self._render_transcript_full("".join(self._history_display))

    def action_focus_transcript(self) -> None:
        self._transcript().focus()

    def action_focus_input(self) -> None:
        self._input().focus()

    def action_submit_input(self) -> None:
        inp = self._input()
        text = (inp.text or "").strip()
        if not text:
            return
        inp.text = ""
        self._resize_input_to_content()
        inp.focus()
        self._send(text)

    def action_copy_mode(self) -> None:
        self.push_screen(self.TranscriptCopyScreen(text=self._get_transcript_plain_text()))

    def action_abort(self) -> None:
        if not self.busy:
            return
        if not self._agent.config.abort_hotkey:
            self._append_error("Abort hotkey disabled. Use /abort_hotkey on to enable.")
            return
        try:
            if getattr(getattr(self._agent, "_abort", None), "event", None) is not None and self._agent._abort.event.is_set():  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        self._append_block("Abort", "Abort requested (Ctrl+C). Waiting for the current step to stop…")
        try:
            self._agent.request_abort()
        except Exception:
            pass

    def action_scroll_up(self) -> None:
        self._transcript().scroll_up()

    def action_scroll_down(self) -> None:
        self._transcript().scroll_down()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            return

    async def on_input_submitted(self, message: InputSubmitted) -> None:
        if message.text:
            self._send(message.text)

    async def on_text_area_changed(self, message: TextArea.Changed) -> None:
        if message.text_area.id == "input":
            self._resize_input_to_content()

    def _handle_slash(self, text: str) -> bool:
        t = text.strip()
        if not t.startswith("/"):
            return False

        parts = t.split()
        cmd = parts[0].lower()
        args = parts[1:]
        rest = t[len(parts[0]) :].strip()

        if cmd in {"/help", "/menu"}:
            self._append_block(
                "Help",
                "\n".join(
                    
                    [   
                        "",
                        "Commands:",
                        "- /help or /menu",
                        "- /workspace [path]",
                        "- /agent_model [name]",
                        "- /tool_call_model [name]",
                        "- /max_agent_steps [int]",
                        "- /mode",
                        "- /mode agent|chat|auto",
                        "- /memory_turn",
                        "- /set_memory_turn [-1|N]",
                        "- /memory_compress_threshold",
                        "- /memory_compress_threshold [int tokens]",
                        "- /compress_memory",
                        "- /calibration_tool",
                        "- /action_delay [seconds]",
                        "- /abort_hotkey [on|off]",
                        "- /log_dir [path]",
                        "- /chat new",
                        "- /chat save [name]",
                        "- /chat list",
                        "- /chat resume <name>",
                        "- /clear_cache",
                        "",
                        "Input:",
                        "- Enter: newline",
                        "- Ctrl+S: send",
                        "- Ctrl+I: focus input",
                        "- Ctrl+L: focus log",
                        "- Ctrl+Q: quit",
                    ]
                ),
            )
            return True

        if cmd == "/clear_cache":
            self._confirm_mode = "clear_cache"
            self._append_block(
                "Confirm",
                "\n".join(
                    [
                        "Clear all logs on disk?",
                        "Type:",
                        "  1 - Cancel",
                        "  2 - Delete logs now",
                    ]
                ),
            )
            return True

        if cmd == "/workspace":
            if not rest:
                self._append_block("Workspace", f"Current workspace path: {self._agent.workspace.source_path}")
                return True
            try:
                self._agent.set_workspace(rest)
                self._save_persisted_settings()
                self._append_block("Workspace", f"Workspace set to: {self._agent.workspace.source_path}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd == "/mode":
            if not rest:
                self._append_block("Mode", f"Current mode: {self._agent.config.run_mode}")
                return True
            try:
                self._agent.set_run_mode(rest)
                self._save_persisted_settings()
                self._append_block("Mode", f"Mode set to: {self._agent.config.run_mode}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/memory_turn", "/memory-turn"}:
            self._append_block("Memory Turn", f"Current memory_turns: {self._agent.config.memory_turns}")
            return True

        if cmd in {"/set_memory_turn", "/set-memory-turn"}:
            if not rest:
                self._append_error("Usage: /set_memory_turn [-1|N]")
                return True
            try:
                self._agent.set_memory_turns(int(rest.strip()))
                self._save_persisted_settings()
                self._append_block("Memory Turn", f"memory_turns set to: {self._agent.config.memory_turns}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/memory_compress_threshold", "/memory-compress-threshold"}:
            if not rest:
                self._append_block(
                    "Memory Compress Threshold",
                    f"Current threshold tokens: {getattr(self._agent.config, 'memory_compress_threshold_tokens', 0)}",
                )
                return True
            try:
                self._agent.set_memory_compress_threshold_tokens(int(rest.strip()))
                self._save_persisted_settings()
                self._append_block(
                    "Memory Compress Threshold",
                    f"Threshold set to: {getattr(self._agent.config, 'memory_compress_threshold_tokens', 0)} tokens",
                )
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/compress_memory", "/compress-memory"}:
            if self.busy:
                self._append_error("Busy; wait for the agent to finish the current turn.")
                return True

            self.busy = True
            self._set_status("Compressing…")

            def sink(ev: Mapping[str, Any]) -> None:
                self.post_message(AgentEvent(ev))

            prev = getattr(self._agent, "_event_sink", None)
            setattr(self._agent, "_event_sink", sink)

            def run_bg() -> None:
                try:
                    self._agent.compress_memory(reason="manual_command(/compress_memory)")
                except KeyboardInterrupt:
                    self.post_message(AgentEvent({"type": "aborted"}))
                except Exception as e:
                    self.post_message(AgentEvent({"type": "error", "text": str(e)}))
                finally:
                    setattr(self._agent, "_event_sink", prev)
                    self.post_message(AgentEvent({"type": "idle"}))

            threading.Thread(target=run_bg, daemon=True).start()
            return True

        if cmd in {"/calibration_tool", "/calibration-tool"}:
            ws_path = str(self._agent.workspace.source_path)
            try:
                subprocess.Popen([sys.executable, "-m", "src.calibrate_gui", "--workspace", ws_path], close_fds=True)
                self._append_block(
                    "Calibration",
                    "\n".join(
                        [
                            f"Launched calibrator for: {ws_path}",
                            "Save in the calibrator, then rerun your command.",
                        ]
                    ),
                )
            except Exception as e:
                self._append_error(f"Failed to launch calibrator: {e}")
            return True

        if cmd in {"/agent_model", "/agent-model", "/model"}:
            if not rest:
                self._append_block("Agent Model", f"Current agent model: {self._agent.config.agent_model}")
                return True
            try:
                self._agent.set_agent_model(rest)
                self._save_persisted_settings()
                self._append_block("Agent Model", f"Agent model set to: {self._agent.config.agent_model}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/tool_call_model", "/tool-call-model"}:
            if not rest:
                self._append_block("Tool Call Model", f"Current tool-call model: {self._agent.config.tool_call_model}")
                return True
            try:
                self._agent.set_tool_call_model(rest)
                self._save_persisted_settings()
                self._append_block("Tool Call Model", f"Tool-call model set to: {self._agent.config.tool_call_model}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/max_agent_steps", "/max-agent-steps"}:
            if not rest:
                self._append_block("Max Agent Steps", f"Current max agent steps: {self._agent.config.max_steps}")
                return True
            try:
                self._agent.set_max_steps(int(rest))
                self._save_persisted_settings()
                self._append_block("Max Agent Steps", f"Max agent steps set to: {self._agent.config.max_steps}")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/action_delay", "/action-delay"}:
            if not rest:
                self._append_block("Action Delay", f"Current action delay: {self._agent.config.action_delay_s:g} seconds")
                return True
            try:
                self._agent.set_action_delay_s(float(rest))
                self._save_persisted_settings()
                self._append_block("Action Delay", f"Action delay set to: {self._agent.config.action_delay_s:g} seconds")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/abort_hotkey", "/abort-hotkey"}:
            if not rest:
                status = "ON" if self._agent.config.abort_hotkey else "OFF"
                self._append_block("Abort Hotkey", f"The abort hotkey status is currently: {status}.")
                return True
            val = rest.strip().lower()
            truthy = {"1", "true", "on", "yes", "enable", "enabled"}
            falsy = {"0", "false", "off", "no", "disable", "disabled"}
            if val not in truthy and val not in falsy:
                self._append_error("Usage: /abort_hotkey [on|off]")
                return True
            try:
                self._agent.set_abort_hotkey(val in truthy)
                self._save_persisted_settings()
                status = "ON" if self._agent.config.abort_hotkey else "OFF"
                self._append_block("Abort Hotkey", f"The abort hotkey status is set to be: {status}.")
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd in {"/log_dir", "/log-dir"}:
            if not rest:
                self._append_block("Log Dir", f"Current log dir: {self._agent.config.log_dir}")
                return True
            try:
                self._agent.set_log_dir(rest)
                self._save_persisted_settings()
                self._append_block("Log Dir", f"Log dir set to: {self._agent.config.log_dir}")
                self._request_cache_usage_update()
            except Exception as e:
                self._append_error(str(e))
            return True

        if cmd == "/chat":
            if not args:
                self._append_error("Usage: /chat save [name] | /chat list | /chat resume <name>")
                return True
            sub = args[0].lower()
            rest = args[1:]
            if sub == "new":
                self._confirm_mode = "new_session"
                self._append_block(
                    "Confirm",
                    "\n".join(
                        [
                            "Are you sure to start a new session?",
                            "Make sure you save your current session before starting a new session.",
                            "",
                            "Type:",
                            "  1 - Cancel and continue current session",
                            "  2 - Start a new session (without saving)",
                        ]
                    ),
                )
                return True
            if sub == "list":
                sessions = list_sessions()
                if not sessions:
                    self._append_block("Sessions", "(none)")
                else:
                    self._append_block("Sessions", "\n".join(sessions))
                return True
            if sub == "save":
                name = rest[0] if rest else datetime.now().strftime("session_%Y%m%d_%H%M%S")
                history_text = "".join(self._history_full)
                agent_state = self._agent.export_session()
                entries = list(self._history_entries)
                state = {
                    "name": name,
                    "workspace": str(self._agent.workspace.source_path),
                    "history": entries,
                    "history_text": history_text,
                    "log_roots": sorted(self._session_log_roots),
                    "agent": agent_state,
                    "tokens": agent_state.get("tokens", {}),
                }
                save_session(name, state)
                self._current_session_name = name
                self._current_session_saved = True
                self._append_block("Sessions", f"Saved: {name}")
                return True
            if sub == "resume":
                if not rest:
                    self._append_error("Usage: /chat resume <name>")
                    return True
                name = rest[0]
                state = load_session(name)
                if state is None:
                    self._append_error(f"Session not found: {name}")
                    return True
                st_ws = state.get("workspace", None)
                if isinstance(st_ws, str) and st_ws.strip():
                    try:
                        self._agent.set_workspace(st_ws)
                    except Exception as e:
                        self._append_error(f"Failed to load workspace from session: {e}")
                hist_val = state.get("history", "")
                history_text = ""
                if isinstance(hist_val, str):
                    history_text = hist_val
                elif isinstance(hist_val, list):
                    items: list[dict[str, Any]] = [it for it in hist_val if isinstance(it, dict)]
                    history_text = self._serialize_history_items(items)
                    self._history_entries = items
                else:
                    history_text = str(state.get("history_text", ""))

                self._history_full = [history_text]
                self._history_display = [history_text]
                self._rerender_transcript_from_history()
                agent_state = state.get("agent", {})
                if isinstance(agent_state, dict):
                    self._agent.import_session(agent_state)
                st_tokens = state.get("tokens", None)
                if isinstance(st_tokens, dict):
                    try:
                        self._set_tokens(
                            input_tokens=int(st_tokens.get("input_tokens", 0)),
                            output_tokens=int(st_tokens.get("output_tokens", 0)),
                            total_tokens=int(st_tokens.get("total_tokens", 0)),
                        )
                    except Exception:
                        self._sync_tokens_from_agent()
                else:
                    self._sync_tokens_from_agent()
                self._current_session_name = name
                self._current_session_saved = True
                self._session_log_roots = set()
                loaded_roots = state.get("log_roots", None)
                if isinstance(loaded_roots, list):
                    for x in loaded_roots:
                        if isinstance(x, str) and x.strip():
                            self._session_log_roots.add(x.strip())
                self._reset_run_logger()
                self._append_block("Sessions", f"Resumed: {name}")
                return True
            self._append_error("Usage: /chat save [name] | /chat list | /chat resume <name>")
            return True

        self._append_error(f"Unknown command: {cmd} (try /help)")
        return True

    def _send(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.busy:
            self._append_error("Busy; wait for the agent to finish the current turn.")
            return

        if self._confirm_mode == "new_session":
            if text == "1":
                self._confirm_mode = None
                self._append_block("Confirm", "Canceled. Continuing current session.")
                return
            if text == "2":
                self._confirm_mode = None
                if not self._current_session_saved:
                    self._delete_log_roots(self._session_log_roots)
                self._current_session_name = None
                self._current_session_saved = False
                self._reset_run_logger()
                self._request_cache_usage_update()
                # Clear transcript + reset agent memory, without saving.
                self._history_display = []
                self._history_full = []
                self._history_entries = []
                self._transcript().clear()
                self._clear_temp_session()
                try:
                    self._agent.import_session(
                        {
                            "memory": [],
                            "archive_memory": [],
                            "last_action_log": "(none yet)",
                            "scan_wait_pending": False,
                            "scan_wait_used": False,
                            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        }
                    )
                except Exception:
                    pass
                self._set_tokens(input_tokens=0, output_tokens=0, total_tokens=0)
                self._append_system(f"Workspace: {self._agent.workspace.source_path}")
                self._append_system("New session started. Use /chat save to save later; /help for commands.")
                return
            self._append_error("Please type 1 (cancel) or 2 (start new session).")
            return

        if self._confirm_mode == "clear_cache":
            if text == "1":
                self._confirm_mode = None
                self._append_block("Confirm", "Canceled.")
                return
            if text == "2":
                self._confirm_mode = None
                roots = {Path("logs"), Path(self._agent.config.log_dir)}
                removed = 0
                for root in roots:
                    try:
                        if not root.exists() or not root.is_dir():
                            continue
                        for child in root.iterdir():
                            if child.is_dir():
                                shutil.rmtree(child, ignore_errors=True)
                                removed += 1
                    except Exception:
                        continue
                self._session_log_roots = set()
                self._reset_run_logger()
                self._append_block("Cache", f"Deleted {removed} log directorie(s).")
                self._request_cache_usage_update()
                return
            self._append_error("Please type 1 (cancel) or 2 (delete logs).")
            return

        if self._handle_slash(text):
            return

        self.busy = True
        self._set_status("Running…")
        self._append_user(text)

        def sink(ev: Mapping[str, Any]) -> None:
            self.post_message(AgentEvent(ev))

        prev = getattr(self._agent, "_event_sink", None)
        setattr(self._agent, "_event_sink", sink)

        def run_bg() -> None:
            try:
                self._agent.run(user_command=text)
            except KeyboardInterrupt:
                self.post_message(AgentEvent({"type": "aborted"}))
            except Exception as e:
                self.post_message(AgentEvent({"type": "error", "text": str(e)}))
            finally:
                setattr(self._agent, "_event_sink", prev)
                self.post_message(AgentEvent({"type": "idle"}))

        threading.Thread(target=run_bg, daemon=True).start()

    async def on_agent_event(self, message: AgentEvent) -> None:
        ev = message.ev
        t = ev.get("type")
        if t == "tokens":
            try:
                self._set_tokens(
                    input_tokens=int(ev.get("input_tokens", self._tokens_in)),
                    output_tokens=int(ev.get("output_tokens", self._tokens_out)),
                    total_tokens=int(ev.get("total_tokens", self._tokens_total)),
                )
            except Exception:
                pass
            return
        if t == "idle":
            self.busy = False
            self._set_status("Ready")
            self.action_focus_input()
            return
        if t == "error":
            self._append_error(str(ev.get("text", "")))
            self.busy = False
            self._set_status("Ready")
            self.action_focus_input()
            return
        if t == "aborted":
            self._append_block("Aborted", "Stopped.")
            return
        if t == "plan":
            plan = ev.get("plan", None)
            if isinstance(plan, list) and plan:
                self._append_block("Plan", "\n".join(f"- {x}" for x in plan))
            return
        if t == "plan_text":
            txt = str(ev.get("text", "") or "").strip()
            if txt:
                self._append_block("Plan", txt)
            return
        if t == "step_blocks":
            action = str(ev.get("action", "") or "").strip()
            observe = str(ev.get("observe", "") or "").strip()
            think = str(ev.get("think", "") or "").strip()
            nxt = str(ev.get("next", "") or "").strip()
            self._append_block("Action", action or "(none)")
            self._append_block("Observe", observe or "(none)")
            self._append_block("Think", think or "(none)")
            self._append_block("Next", nxt or "(none)")
            return
        if t == "decision":
            say = str(ev.get("say", "")).strip()
            if say:
                self._append_agent(say)
                self._rerender_transcript_from_history()
            plan = ev.get("plan", None)
            if isinstance(plan, list) and plan:
                self._append_block("Plan", "\n".join(f"- {x}" for x in plan))
            obs = str(ev.get("observation", "") or "").strip()
            if obs:
                self._append_block("Observation", obs)
            rat = str(ev.get("rationale", "") or "").strip()
            if rat:
                self._append_block("Thought", rat)
            action = str(ev.get("action", ""))
            action_input = ev.get("action_input", {})
            try:
                ai = json.dumps({"tool": action, "args": action_input}, indent=2, ensure_ascii=False)
            except Exception:
                ai = str({"tool": action, "args": action_input})
            self._append_block("Tool Call", ai)
            return
        if t == "result":
            result = ev.get("result", {})
            if isinstance(result, dict):
                lr = result.get("log_root", None)
                if isinstance(lr, str) and lr.strip():
                    self._track_log_root(lr.strip())
                    self._request_cache_usage_update()
            try:
                rtxt = json.dumps({"tool": str(ev.get("action", "")), "result": result}, indent=2, ensure_ascii=False)
            except Exception:
                rtxt = str({"tool": str(ev.get("action", "")), "result": result})
            self._append_block("Tool Result", rtxt)
            return
        if t == "finish":
            self._append_block("Done", str(ev.get("say", "Finished.")))
            self._rerender_transcript_from_history()
            return
        if t == "memory_compress":
            phase = str(ev.get("phase", "") or "").strip().lower()
            if phase == "start":
                reason = str(ev.get("reason", "") or "").strip()
                before = ev.get("before_tokens", None)
                thr = ev.get("threshold_tokens", None)
                msg = "Memory compression started."
                if reason:
                    msg += f" reason={reason}"
                if before is not None:
                    msg += f" before_tokens={before}"
                if thr is not None:
                    msg += f" threshold={thr}"
                self._append_system(msg)
                self._rerender_transcript_from_history()
                return
            if phase == "error":
                reason = str(ev.get("reason", "") or "").strip()
                err = str(ev.get("error", "") or "").strip()
                msg = "Memory compression failed."
                if reason:
                    msg += f" reason={reason}"
                if err:
                    msg += f" error={err}"
                self._append_error(msg)
                self._rerender_transcript_from_history()
                return
            if phase == "done":
                reason = str(ev.get("reason", "") or "").strip()
                before = ev.get("before_tokens", None)
                after = ev.get("after_tokens", None)
                arch = ev.get("archive_entries", None)
                memn = ev.get("memory_entries", None)
                msg = "Memory compression finished."
                if reason:
                    msg += f" reason={reason}"
                if before is not None and after is not None:
                    msg += f" tokens={before}->{after}"
                if memn is not None and arch is not None:
                    msg += f" entries(memory={memn}, archive={arch})"
                self._append_system(msg)
                self._rerender_transcript_from_history()
                return
        if t == "chat":
            txt = str(ev.get("text", "") or "").strip()
            if txt:
                self._append_agent(txt)
                self._rerender_transcript_from_history()
            return


def run_chat_tui(*, agent: VisualAutomationAgent, first_message: Optional[str] = None) -> int:
    ChatApp(agent=agent, first_message=first_message).run()
    return 0
