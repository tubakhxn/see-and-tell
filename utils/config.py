"""Application configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Centralized configuration container."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    sample_interval_ms: int = 750
    window_name: str = "SmolVLM Live"
    model_name: str = "Salesforce/blip-image-captioning-base"
    smoothing_window: int = 3
    confidence_threshold: float = 0.35
    max_caption_tokens: int = 60
    device: str = "auto"  # "cuda", "cpu", or "auto"
    inference_short_side: int = 512
    worker_idle_sleep_ms: int = 15


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def load_config() -> AppConfig:
    """Create configuration overriding defaults via env vars when provided."""

    def _int(key: str, default: int) -> int:
        try:
            return int(_env(key, str(default)))
        except ValueError:
            return default

    def _float(key: str, default: float) -> float:
        try:
            return float(_env(key, str(default)))
        except ValueError:
            return default

    return AppConfig(
        camera_index=_int("APP_CAMERA_INDEX", AppConfig.camera_index),
        frame_width=_int("APP_FRAME_WIDTH", AppConfig.frame_width),
        frame_height=_int("APP_FRAME_HEIGHT", AppConfig.frame_height),
        sample_interval_ms=_int("APP_SAMPLE_INTERVAL", AppConfig.sample_interval_ms),
        window_name=_env("APP_WINDOW_NAME", AppConfig.window_name),
        model_name=_env("APP_MODEL_NAME", AppConfig.model_name),
        smoothing_window=_int("APP_SMOOTHING_WINDOW", AppConfig.smoothing_window),
        confidence_threshold=_float(
            "APP_CONFIDENCE_THRESHOLD", AppConfig.confidence_threshold
        ),
        max_caption_tokens=_int(
            "APP_MAX_CAPTION_TOKENS", AppConfig.max_caption_tokens
        ),
        device=_env("APP_DEVICE", AppConfig.device),
        inference_short_side=_int(
            "APP_INFERENCE_SHORT_SIDE", AppConfig.inference_short_side
        ),
        worker_idle_sleep_ms=_int(
            "APP_WORKER_IDLE_SLEEP_MS", AppConfig.worker_idle_sleep_ms
        ),
    )
