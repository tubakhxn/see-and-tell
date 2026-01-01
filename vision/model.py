"""Vision-language model wrapper."""
from __future__ import annotations

import threading
from typing import Optional

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionLanguageModel:
    """Loads a BLIP captioning model and exposes a describe API."""

    def __init__(self, model_name: str, device_preference: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._select_device(device_preference)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    @staticmethod
    def _select_device(pref: str) -> torch.device:
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def describe(self, image, max_tokens: int = 60) -> str:
        """Generate a natural language description for a PIL image."""
        with self._lock:
            with torch.inference_mode():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=3,
                    repetition_penalty=1.1,
                )
        caption = self.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        return caption.strip()
