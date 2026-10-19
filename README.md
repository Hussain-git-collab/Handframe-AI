# HandFrame AI (modernized)

Make a floating frame with both hands, pinch to capture, and restyle the capture in 17 artistic styles, including a character-based ASCII Terminal effect.

This version uses the **MediaPipe Tasks `HandLandmarker` API** (the old `mp.solutions.hands` API is removed
in current MediaPipe releases) and works on current Python / OpenCV / NumPy.

## Install
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
The hand model (~8 MB) downloads automatically to `models/` on first run
(or pass `--hand_model path/to/hand_landmarker.task`).

## Controls
| Action | Effect |
|---|---|
| Make an "L" with both hands (thumb + index of each hand = 4 corners) | Floating frame appears |
| Pinch (thumb to index, either hand) | Capture what is inside the frame |
| `[` / `]` (or `A` / `D`) | Previous / next style |
| `S` | Save current result to `captures/` |
| `R` | Reset capture |
| `Q` / `Esc` | Quit |

After a capture, the styled image floats inside your frame with perspective; drop your hands and it
moves to a corner thumbnail.

## Faculty demo checklist

1. Use good front lighting and keep both hands fully inside the camera view.
2. Form a large rectangle with the thumb and index finger of each hand, then pause briefly for the amber frame to stabilize.
3. Pinch with either hand to take the perspective-corrected capture. Use `[` and `]` to show fast, local image-processing styles; save a result with `S`.
4. If a webcam is unavailable or you want a rehearsed presentation, record a short hand-gesture clip and use `python app.py --video demo.mp4 --loop-video`. Add `--no-mirror` for a pre-mirrored recording.

## What this demonstrates

- Real-time hand landmark detection with MediaPipe Tasks
- Landmark smoothing (One Euro filter) and hysteresis-based pinch detection
- Convex quadrilateral validation, perspective rectification, and inverse perspective projection
- Asynchronous image processing with 17 OpenCV artistic filters; optional FLUX image-to-image generation
- Responsive interaction: stale image-generation work is discarded when a new capture is taken

## Verification

Run the repeatable non-camera checks before presenting:

```bash
python -m unittest discover -s tests -v
```

## Optional: real AI styles with FLUX.1
```bash
pip install torch diffusers transformers accelerate sentencepiece protobuf pillow
python app.py --flux                       # downloads FLUX.1-schnell from Hugging Face
python app.py --model_path /path/to/flux   # local model
```
Generation runs on a background thread, so the video never freezes. If FLUX fails to load,
the app falls back to the instant OpenCV styles. FLUX needs a strong GPU (use `--flux_size 384` if short on VRAM).

## Options
`--camera N`, `--video PATH`, `--loop-video`, `--width`, `--height`, `--no-mirror`, `--flux`, `--model_path`, `--hand_model`, `--flux_size`, `--flux_steps`
