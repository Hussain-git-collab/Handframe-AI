"""HandFrame AI - make a frame with both hands, pinch to capture, restyle instantly."""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import HandTracker
from inference import StyleEngine
from perspective import order_quad, quad_valid, unwarp, warp_into
from smoothing import OneEuroFilter
from styles import STYLES
from ui_overlay import draw_hand, draw_hud, draw_quad, draw_thumbnail, flash, text


def parse_args():
    p = argparse.ArgumentParser(description="HandFrame AI")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="optional recorded video source for a repeatable demo")
    p.add_argument("--loop-video", action="store_true", help="loop --video continuously during a presentation")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="disable selfie-view mirroring")
    p.add_argument("--flux", action="store_true", help="use FLUX.1 img2img (needs torch + diffusers)")
    p.add_argument("--model_path", default=None, help="local/HF path of a FLUX model")
    p.add_argument("--hand_model", default=None, help="path to hand_landmarker.task")
    p.add_argument("--flux_size", type=int, default=512)
    p.add_argument("--flux_steps", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.loop_video and not args.video:
        raise SystemExit("--loop-video requires --video PATH")
    if args.video and not Path(args.video).is_file():
        raise SystemExit(f"Video file not found: {args.video}")
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        cap.release()
        source_name = f"video '{args.video}'" if args.video else f"camera {args.camera}"
        raise SystemExit(f"Cannot open {source_name}")

    tracker = HandTracker(model_path=args.hand_model)
    engine = StyleEngine(use_flux=args.flux or bool(args.model_path), model_path=args.model_path,
                         size=args.flux_size, steps=args.flux_steps)

    qfilter = OneEuroFilter(min_cutoff=1.2, beta=0.01)
    quad_hist = deque(maxlen=8)          # recent clean quads (used when fingers close to pinch)
    quad, quad_t = None, 0.0
    capture, capture_id, style_idx = None, 0, 0
    pinch_prev, last_capture_t, flash_amt = False, 0.0, 0.0
    fps, t_prev = 0.0, time.monotonic()
    notice, notice_t = "", 0.0

    win = "HandFrame AI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def toast(msg):
        nonlocal notice, notice_t
        notice, notice_t = msg, time.monotonic()

    def set_style(i):
        nonlocal style_idx
        style_idx = i % len(STYLES)
        if capture is not None:
            engine.request(capture_id, capture, style_idx)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video and args.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pinch_prev = False
                    quad, quad_t = None, 0.0
                    qfilter.reset()
                    quad_hist.clear()
                    toast("Replaying demo video")
                    continue
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            clean = frame.copy()
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

            hands = tracker.detect(frame)

        # ---- frame from two hands: thumb + index tips of each hand = 4 corners ----
            any_pinch = any(h.pinching for h in hands)
            if len(hands) == 2:
                if any_pinch and quad_hist:
                    quad = quad_hist[-1]                # hold the newest stable pre-pinch frame
                else:
                    raw = order_quad([hands[0].thumb_tip, hands[0].index_tip,
                                      hands[1].thumb_tip, hands[1].index_tip])
                    if quad_valid(raw, frame.shape):
                        quad = qfilter(raw.ravel(), now).reshape(4, 2).astype(np.float32)
                        quad_hist.append(quad.copy())
                        quad_t = now
            elif now - quad_t > 0.3:
                quad = None
                qfilter.reset()
                quad_hist.clear()

        # ---- pinch-to-capture (rising edge, with cooldown) ----
            if (any_pinch and not pinch_prev and len(hands) == 2 and quad_hist
                    and now - last_capture_t > 1.0):
                q = quad_hist[-1]
                if capture is not None:
                    engine.drop_capture(capture_id)
                capture = unwarp(clean, q)
                capture_id += 1
                engine.request(capture_id, capture, style_idx)
                last_capture_t, flash_amt = now, 0.8
                toast("Captured!")
            pinch_prev = any_pinch

        # ---- compose ----
            display = frame
            panel = None
            busy = False
            if capture is not None:
                result = engine.get(capture_id, style_idx)
                busy = result is None
                if busy:
                    engine.request(capture_id, capture, style_idx)
                panel = capture if result is None else result
                if busy:
                    panel = (panel * 0.5).astype(np.uint8)

            if quad is not None and panel is not None:
                display = warp_into(display, panel, quad)
            elif quad is not None:
                draw_quad(display, quad)
                text(display, "pinch to capture", (int(quad[0][0]), max(int(quad[0][1]) - 12, 20)), 0.6, (255, 255, 255))
            elif panel is not None:
                draw_thumbnail(display, panel)

            for h in hands:
                draw_hand(display, h)
            display = flash(display, flash_amt)
            flash_amt = max(0.0, flash_amt - 0.1)

            hint = "Two L-hands = frame | pinch = capture | [ ] style | S save | R reset | Q quit"
            if now - notice_t < 1.5:
                hint = notice
            draw_hud(display, fps, STYLES[style_idx].name, style_idx, len(STYLES),
                     engine.status, hint, busy)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key in (ord("]"), ord("d")):
                set_style(style_idx + 1)
            elif key in (ord("["), ord("a")):
                set_style(style_idx - 1)
            elif key == ord("r"):
                if capture is not None:
                    engine.drop_capture(capture_id)
                capture = None
                toast("Reset")
            elif key == ord("s") and capture is not None:
                result = engine.get(capture_id, style_idx)
                if result is not None:
                    out_dir = Path("captures")
                    out_dir.mkdir(exist_ok=True)
                    name = out_dir / f"{int(time.time())}_{STYLES[style_idx].name.replace(' ', '_')}.png"
                    cv2.imwrite(str(name), result)
                    toast(f"Saved {name}")
    finally:
        cap.release()
        tracker.close()
        engine.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
