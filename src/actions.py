from __future__ import annotations

import time
from dataclasses import dataclass

import pyautogui

from .workspace import Anchor
from threading import Event


@dataclass(frozen=True, slots=True)
class ActionConfig:
    delay_s: float = 0.25


class Actor:
    def __init__(self, *, dry_run: bool, config: ActionConfig, abort_event: Event | None = None) -> None:
        self._dry_run = dry_run
        self._config = config
        self._abort_event = abort_event

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

    def _check_abort(self) -> None:
        if self._abort_event is not None and self._abort_event.is_set():
            raise KeyboardInterrupt()

    def sleep(self) -> None:
        if self._config.delay_s > 0:
            time.sleep(self._config.delay_s)

    def click(self, anchor: Anchor) -> None:
        self._check_abort()
        if self._dry_run:
            return
        x, y = anchor.as_point()
        pyautogui.click(x=x, y=y)
        self.sleep()

    def double_click(self, anchor: Anchor) -> None:
        self._check_abort()
        if self._dry_run:
            return
        x, y = anchor.as_point()
        pyautogui.doubleClick(x=x, y=y, interval=0.05)
        self.sleep()

    def hotkey(self, *keys: str) -> None:
        self._check_abort()
        if self._dry_run:
            return
        pyautogui.hotkey(*keys)
        self.sleep()

    def press(self, key: str) -> None:
        self._check_abort()
        if self._dry_run:
            return
        pyautogui.press(key)
        self.sleep()

    def type_text(self, text: str) -> None:
        self._check_abort()
        if self._dry_run:
            return
        pyautogui.write(text, interval=0.01)
        self.sleep()
