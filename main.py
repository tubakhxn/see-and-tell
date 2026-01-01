"""Entry point for the SmolVLM real-time webcam demo."""
from __future__ import annotations

import sys
import textwrap
from typing import List

import cv2

from camera.webcam import WebcamStream
from utils.config import load_config
from utils.fps import FPSCounter
from vision.inference import FrameAnalyzer
from vision.model import VisionLanguageModel


def draw_panel(
    frame,
    caption: str,
    objects: List[str],
    actions: List[str],
    fps_value: float,
) -> None:
    """Render textual diagnostics on top of the frame."""
    h, w, _ = frame.shape
    panel_width = int(w * 0.45)
    panel_height = int(h * 0.35)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        color=(0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS: {fps_value:5.1f}",
        "Description:",
    ]
    wrapped_caption = textwrap.wrap(caption, width=46) or ["..."]
    lines.extend(wrapped_caption)
    objects_text = ", ".join(objects) if objects else "-"
    actions_text = ", ".join(actions) if actions else "-"
    lines.append(f"Objects: {objects_text}")
    lines.append(f"Actions: {actions_text}")

    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


def main() -> int:
    config = load_config()
    print("Loading vision-language model...", flush=True)
    model = VisionLanguageModel(config.model_name, config.device)
    analyzer = FrameAnalyzer(config, model)
    fps = FPSCounter()

    try:
        with WebcamStream(
            index=config.camera_index,
            width=config.frame_width,
            height=config.frame_height,
        ) as stream:
            print("Press 'q' or ESC to exit.")
            while True:
                frame = stream.read()
                if frame is None:
                    continue

                desc = analyzer.analyze(frame)
                fps_value = fps.tick()

                draw_panel(
                    frame,
                    caption=desc.caption,
                    objects=desc.objects,
                    actions=desc.actions,
                    fps_value=fps_value,
                )

                cv2.imshow(config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except RuntimeError as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        analyzer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
