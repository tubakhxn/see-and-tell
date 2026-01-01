"""High-level frame analysis pipeline."""
from __future__ import annotations

import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import threading

import cv2
from PIL import Image

from utils.config import AppConfig
from vision.model import VisionLanguageModel


class DescriptionSmoother:
    """Stabilizes noisy captions by majority voting across a window."""

    def __init__(self, window: int) -> None:
        self.window = max(1, window)
        self.history: Deque[str] = deque(maxlen=self.window)

    def push(self, caption: str) -> str:
        self.history.append(caption)
        if len(self.history) == 1:
            return caption
        counts = Counter(self.history)
        most_common = counts.most_common(1)[0][0]
        return most_common if counts[most_common] > 1 else self.history[-1]


class ObjectActionExtractor:
    """Extract coarse objects/actions mentioned inside captions."""

    OBJECTS = {
        "bottle",
        "can",
        "phone",
        "book",
        "cup",
        "mug",
        "laptop",
        "keyboard",
        "mouse",
        "remote",
        "controller",
        "bag",
        "watch",
        "glasses",
        "pen",
        "pencil",
        "tablet",
    }
    ACTIONS = {
        "holding",
        "showing",
        "pointing",
        "drinking",
        "typing",
        "reading",
        "smiling",
        "talking",
        "sitting",
        "standing",
        "looking",
        "presenting",
    }

    def extract(self, caption: str) -> Tuple[List[str], List[str]]:
        text = caption.lower()
        tokens = set(re.findall(r"[a-zA-Z]+", text))
        objects = sorted(token for token in self.OBJECTS if token in tokens)
        actions = sorted(token for token in self.ACTIONS if token in tokens)
        return objects, actions


@dataclass
class FrameDescription:
    caption: str
    objects: List[str]
    actions: List[str]
    timestamp: float


class FrameAnalyzer:
    """Coordinates model inference, smoothing, and metadata extraction."""

    def __init__(self, config: AppConfig, model: VisionLanguageModel) -> None:
        self.config = config
        self.model = model
        self.smoother = DescriptionSmoother(config.smoothing_window)
        self.extractor = ObjectActionExtractor()
        self._last_sample = 0.0
        self._state = FrameDescription(
            caption="Initializing vision model...",
            objects=[],
            actions=[],
            timestamp=time.perf_counter(),
        )
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[object] = None
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _idle(self) -> None:
        time.sleep(max(1, self.config.worker_idle_sleep_ms) / 1000.0)

    def _should_sample(self, now: float) -> bool:
        interval = max(0.05, self.config.sample_interval_ms / 1000.0)
        return now - self._last_sample >= interval

    def _downscale(self, frame):
        target = max(64, self.config.inference_short_side)
        h, w = frame.shape[:2]
        short = min(h, w)
        if short <= target:
            return frame
        scale = target / short
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _to_pil(self, frame) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _process(self, frame, now: float) -> None:
        pil_image = self._to_pil(self._downscale(frame))
        caption = self.model.describe(
            pil_image, max_tokens=self.config.max_caption_tokens
        )
        smoothed = self.smoother.push(caption)
        objects, actions = self.extractor.extract(smoothed)
        self._state = FrameDescription(
            caption=smoothed,
            objects=objects,
            actions=actions,
            timestamp=now,
        )
        self._last_sample = now

    def _worker_loop(self) -> None:
        while self._running:
            now = time.perf_counter()
            if not self._should_sample(now):
                self._idle()
                continue
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame = None
            if frame is None:
                self._idle()
                continue
            try:
                self._process(frame, now)
            except Exception:
                # Keep the worker alive even if inference fails once.
                self._idle()

    def analyze(self, frame) -> FrameDescription:
        if frame is not None:
            with self._frame_lock:
                self._latest_frame = frame.copy()
        return self._state

    def latest(self) -> FrameDescription:
        return self._state

    def close(self) -> None:
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
