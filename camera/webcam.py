"""Threaded webcam capture helper."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2


class WebcamStream:
    """Continuously grabs frames on a background thread for low latency."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {index}")

        # Configure camera output size when supported.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        self._frame_lock = threading.Lock()
        self._frame: Optional[cv2.typing.MatLike] = None
        self._running = False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)

    def start(self) -> "WebcamStream":
        if self._running:
            return self
        self._running = True
        self._thread.start()
        return self

    def _update_loop(self) -> None:
        while self._running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._frame = frame

    def read(self) -> Optional[cv2.typing.MatLike]:
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.release()

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()
