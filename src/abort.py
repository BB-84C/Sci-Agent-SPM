from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Optional


@dataclass(frozen=True, slots=True)
class AbortHandle:
    event: Event
    _listener: Optional[object]

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass


def start_abort_hotkey(*, key: str = "esc") -> AbortHandle:
    event = Event()

    try:
        from pynput import keyboard

        target = keyboard.Key.esc if key.lower() == "esc" else None

        def on_press(k: object) -> None:
            if target is not None and k == target:
                event.set()

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        return AbortHandle(event=event, _listener=listener)
    except Exception:
        return AbortHandle(event=event, _listener=None)

