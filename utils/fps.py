"""Simple FPS counter utility."""
from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Tracks instantaneous and smoothed frames-per-second."""

    def __init__(self, history: int = 64) -> None:
        self.history: Deque[float] = deque(maxlen=history)
        self.last = time.perf_counter()
        self._fps = 0.0
        self.frames = 0

    def tick(self) -> float:
        """Record a frame boundary and return the latest FPS value."""
        now = time.perf_counter()
        delta = now - self.last
        self.last = now
        if delta > 0:
            fps = 1.0 / delta
            self.history.append(fps)
            self._fps = sum(self.history) / len(self.history)
        self.frames += 1
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps
