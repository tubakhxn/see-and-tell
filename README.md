
# SmolVLM Real-Time Webcam Demo

**ForkingInString:** I use OpenCV and Hugging Face BLIP to turn webcam video into instant live captions and object detection.

**Creator/Dev:** [dev-tubakhxn]

Local-first real-time webcam experience inspired by HuggingFace SmolVLM demos. The app continuously captures frames, runs a lightweight BLIP vision-language model, and overlays natural language descriptions plus detected handheld objects/actions on screen.

## Features
- Live webcam capture with threaded frame reader for smooth FPS
- Configurable sampling interval so inference does not block rendering
- BLIP captioning model (runs on CPU or GPU) for natural-language descriptions
- Sliding-window smoothing to reduce caption flicker
- Keyword-based object/action extraction for quick status badges
- Overlay HUD renders description, detected objects, actions, and FPS
- Graceful shutdown on `q` or `ESC`

## Project Layout
```
main.py
camera/
  webcam.py          # threaded OpenCV capture helper
vision/
  model.py           # Hugging Face BLIP wrapper
  inference.py       # sampling, smoothing, keyword extraction
utils/
  config.py          # environment-aware configuration
  fps.py             # moving-average FPS counter
requirements.txt
README.md
```

## Prerequisites
- Python 3.10+
- Webcam accessible by OpenCV
- (Optional) CUDA-compatible GPU for faster captions

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate            # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** The first BLIP download happens automatically the first time you run the app and is cached under `~/.cache/huggingface`.

## Running the App
```bash
python main.py
```
Press `q` or `ESC` to exit. The overlay displays FPS, current caption, detected objects, and actions.

## Configuration
Tune behavior via environment variables (defaults in `utils/config.py`):

| Variable | Description | Default |
| --- | --- | --- |
| `APP_CAMERA_INDEX` | Webcam index passed to OpenCV | `0` |
| `APP_FRAME_WIDTH` / `APP_FRAME_HEIGHT` | Requested capture size | `1280` / `720` |
| `APP_SAMPLE_INTERVAL` | Milliseconds between VLM inferences | `750` |
| `APP_SMOOTHING_WINDOW` | Caption smoothing window size | `3` |
| `APP_MODEL_NAME` | Hugging Face model identifier | `Salesforce/blip-image-captioning-base` |
| `APP_DEVICE` | `cpu`, `cuda`, or `auto` | `auto` |
| `APP_MAX_CAPTION_TOKENS` | Max new tokens during generation | `60` |

Example (PowerShell):
```powershell
$env:APP_SAMPLE_INTERVAL=500
$env:APP_DEVICE="cuda"
python main.py
```

## Troubleshooting
- **Camera busy / cannot open:** Ensure no other app uses the webcam, or change `APP_CAMERA_INDEX`.
- **Slow captions on CPU:** Increase `APP_SAMPLE_INTERVAL` to sample less frequently.
- **Model download errors:** Run `huggingface-cli login` if the model requires authentication or retry when the network is stable.

## Extending
- Swap `APP_MODEL_NAME` for another captioning/VLM checkpoint that Hugging Face Transformers supports.
- Replace `ObjectActionExtractor` with a more advanced parser or detector (e.g., Grounding DINO or open-vocabulary detectors) without touching the UI loop.
