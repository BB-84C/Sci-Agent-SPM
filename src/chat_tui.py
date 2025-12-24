from __future__ import annotations

import json
import re
import threading
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

from rich.text import Text

from .agent import VisualAutomationAgent
from .session_store import list_sessions, load_session, save_session


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
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "focus_transcript", "Log"),
        ("ctrl+i", "focus_input", "Input"),
        ("ctrl+shift+c", "copy_mode", "Select/Copy"),
        ("pageup", "scroll_up", "Up"),
        ("pagedown", "scroll_down", "Down"),
    ]

    busy: bool = reactive(False)

    def __init__(self, *, agent: VisualAutomationAgent, first_message: Optional[str] = None) -> None:
        super().__init__()
        self._agent = agent
        self._first_message = first_message.strip() if first_message else None
        self._theme = Theme()
        self._history: list[str] = []
        self._events: list[Mapping[str, Any]] = []
        self._confirm_mode: Optional[str] = None
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._tokens_total: int = 0
        self._truncate_tool_blocks: bool = True

    class ChatInput(TextArea):
        ALLOW_SELECT = True
        # Override TextArea defaults so selection/copy works like a normal editor.
        BINDINGS = [
            Binding("ctrl+a", "select_all", "Select all", show=True, priority=True),
            Binding("ctrl+c", "copy", "Copy", show=True, priority=True),
        ]

        async def _on_key(self, event: events.Key) -> None:
            # TextArea inserts newlines in its internal _on_key(). Intercept here so:
            # - Enter sends
            # - Shift+Enter inserts newline
            key = (event.key or "").lower()
            if key == "shift+enter":
                event.stop()
                event.prevent_default()
                self.insert("\n")
                return
            if key == "enter":
                event.stop()
                event.prevent_default()
                text = self.text.strip()
                self.text = ""
                self.post_message(InputSubmitted(text))
                return
            await super()._on_key(event)

    class Transcript(RichLog):
        BINDINGS = [
            Binding("ctrl+c", "copy_transcript", "Copy transcript", show=False, priority=True),
        ]

        def action_copy_transcript(self) -> None:
            app = getattr(self, "app", None)
            if app is None:
                return
            try:
                app.copy_to_clipboard(getattr(app, "_get_transcript_plain_text")())
            except Exception:
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
                yield Static("", id="underleft")
                yield Static("", id="tokens")

            yield Footer()

    def on_mount(self) -> None:
        self._set_status("Ready")
        self._append_system(f"Workspace: {self._agent.workspace.source_path}")
        self._append_system("Enter: send • Shift+Enter: newline • /help for commands")
        self._append_system("Safety: ESC abort hotkey • Mouse to top-left FAILSAFE")
        self._sync_tokens_from_agent()
        inp = self.query_one("#input", TextArea)
        inp.focus()
        self._resize_input_to_content()
        if self._first_message:
            self.call_later(lambda: self._send(self._first_message or ""))

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

        # Use the actual content width to estimate wrapped lines.
        width = int(getattr(inp.content_size, "width", 0))
        if width <= 0:
            width = max(20, int(getattr(self.size, "width", 120)) - 12)

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

    def _append_raw(self, text: str) -> None:
        self._history.append(text)
        self._render_transcript_chunk(text)

    def _append_system(self, text: str) -> None:
        self._append_raw(f"[SYSTEM] {text}\n\n")

    def _append_user(self, text: str) -> None:
        self._append_raw(f"[YOU] {text}\n\n")

    def _append_agent(self, text: str) -> None:
        self._append_raw(f"[AGENT] {text}\n\n")

    def _append_block(self, title: str, content: str) -> None:
        self._append_raw(f"[{title.upper()}]\n{content}\n\n")

    def _append_error(self, text: str) -> None:
        self._append_raw(f"[ERROR] {text}\n\n")

    def _get_transcript_plain_text(self) -> str:
        return "".join(self._history)

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

        def extract_tool_from_tool_call(content: str) -> Optional[str]:
            for line in (content or "").splitlines():
                m = re.match(r"\\s*tool\\s*:\\s*(.+?)\\s*$", line, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip() or None
            try:
                obj = json.loads(content)
            except Exception:
                return None
            if isinstance(obj, dict):
                for k in ("tool", "name", "skill", "action"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return None

        def extract_tool_or_skill_from_result(content: str) -> Optional[tuple[str, str]]:
            try:
                obj = json.loads(content)
            except Exception:
                return None
            if not isinstance(obj, dict):
                return None
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

        out: list[TranscriptEntry] = []
        for idx, entry in enumerate(entries):
            tag = entry.tag.strip().upper()
            if idx < last_response_idx and tag in {"TOOL CALL", "TOOL RESULT"}:
                if tag == "TOOL CALL":
                    tool = extract_tool_from_tool_call(entry.content or "")
                    collapsed = f"tool: {tool}" if tool else "tool: (truncated)"
                    out.append(TranscriptEntry(tag=entry.tag, content=collapsed))
                else:
                    payload = extract_tool_or_skill_from_result(entry.content or "")
                    if payload is not None:
                        label, name = payload
                        collapsed = f"{label}: {name}"
                    else:
                        tool = tool_name_for_result_at(idx)
                        collapsed = f"tool: {tool}" if tool else "tool: (truncated)"
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

    def _render_entry(self, entry: TranscriptEntry) -> Text:
        t = self._theme
        tag = entry.tag.strip().upper()
        content = entry.content or ""

        def chip(label: str, color: str) -> Text:
            return Text(f" {label} ", style=f"bold {t.bg} on {color}")

        def box_title(label: str, color: str) -> Text:
            return Text(f" {label} ", style=f"bold {t.bg} on {color}")

        if tag == "YOU":
            msg_bg = t.mix(t.user, 0.10)
            msg = Text()
            msg.append_text(chip("YOU", t.user))
            msg.append(" ")
            msg.append(content, style=f"{t.user} on {msg_bg}")
            return msg

        if tag == "AGENT":
            msg_bg = t.mix(t.agent, 0.10)
            msg = Text()
            msg.append_text(chip("AGENT", t.agent))
            msg.append(" ")
            msg.append(content, style=f"{t.fg} on {msg_bg}")
            return msg

        if tag == "SYSTEM":
            msg_bg = t.mix(t.dim, 0.08)
            msg = Text()
            msg.append_text(chip("SYSTEM", t.dim))
            msg.append(" ")
            msg.append(content, style=f"{t.dim} on {msg_bg}")
            return msg

        if tag == "ERROR":
            msg_bg = t.mix(t.error, 0.08)
            msg = Text()
            msg.append_text(chip("ERROR", t.error))
            msg.append(" ")
            msg.append(content, style=f"{t.error} on {msg_bg}")
            return msg

        block_styles: dict[str, tuple[str, str]] = {
            "PLAN": ("PLAN", t.accent),
            "OBSERVATION": ("OBSERVATION", t.dim),
            "THOUGHT": ("THOUGHT", t.agent),
            "TOOL CALL": ("TOOL CALL", t.accent),
            "TOOL RESULT": ("TOOL RESULT", t.user),
            "DONE": ("DONE", t.fg),
            "SESSIONS": ("SESSIONS", t.user),
            "HELP": ("HELP", t.dim),
            "CONFIRM": ("CONFIRM", t.accent),
        }
        if tag in block_styles:
            label, color = block_styles[tag]
            body_bg = t.mix(color, 0.08)
            out = Text()
            out.append_text(box_title(label, color))
            out.append("\n")
            out.append(content, style=f"{t.fg} on {body_bg}")
            return out

        # Fallback: show untagged lines without extra styling.
        return Text(content, style=t.fg)

    def _rerender_transcript_from_history(self) -> None:
        self._render_transcript_full("".join(self._history))

    def action_focus_transcript(self) -> None:
        self._transcript().focus()

    def action_focus_input(self) -> None:
        self._input().focus()

    def action_copy_mode(self) -> None:
        self.push_screen(self.TranscriptCopyScreen(text=self._get_transcript_plain_text()))

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

        if cmd in {"/help", "/menu"}:
            self._append_block(
                "Help",
                "\n".join(
                    [
                        "/help or /menu",
                        "/chat new",
                        "/chat save [name]",
                        "/chat list",
                        "/chat resume <name>",
                        "",
                        "Input:",
                        "- Enter: send",
                        "- Shift+Enter: newline",
                        "- Ctrl+I: focus input",
                        "- Ctrl+L: focus log",
                        "- Ctrl+Q: quit",
                    ]
                ),
            )
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
                agent_state = self._agent.export_session()
                state = {
                    "name": name,
                    "workspace": str(self._agent.workspace.source_path),
                    "history": "".join(self._history),
                    "agent": agent_state,
                    "tokens": agent_state.get("tokens", {}),
                }
                save_session(name, state)
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
                self._history = [str(state.get("history", ""))]
                self._render_transcript_full("".join(self._history))
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
                # Clear transcript + reset agent memory, without saving.
                self._history = []
                self._transcript().clear()
                try:
                    self._agent.import_session(
                        {
                            "memory": [],
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
            return
        if t == "error":
            self._append_error(str(ev.get("text", "")))
            self.busy = False
            self._set_status("Ready")
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
                ai = json.dumps(action_input, indent=2)
            except Exception:
                ai = str(action_input)
            self._append_block("Tool Call", f"tool: {action}\nargs:\n{ai}")
            return
        if t == "result":
            result = ev.get("result", {})
            try:
                rtxt = json.dumps(result, indent=2)
            except Exception:
                rtxt = str(result)
            self._append_block("Tool Result", rtxt)
            return
        if t == "finish":
            self._append_block("Done", str(ev.get("say", "Finished.")))
            self._rerender_transcript_from_history()
            return


def run_chat_tui(*, agent: VisualAutomationAgent, first_message: Optional[str] = None) -> int:
    ChatApp(agent=agent, first_message=first_message).run()
    return 0
