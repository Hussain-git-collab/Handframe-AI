"""Hand tracking + pinch detection using the MediaPipe Tasks HandLandmarker API.

(The legacy `mp.solutions.hands` API was removed in recent MediaPipe releases.)
"""
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from smoothing import OneEuroFilter

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model(path=None) -> Path:
    path = Path(path) if path else DEFAULT_MODEL
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HandFrame] Downloading hand landmarker model -> {path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Could not download the model ({e}).\nDownload it manually from:\n  {MODEL_URL}\n"
            f"and save it as: {path}") from e
    return path


@dataclass
class Hand:
    label: str
    landmarks: np.ndarray      # (21, 2) pixel coords (smoothed)
    pinch_ratio: float         # thumb-index distance / palm size
    pinching: bool

    @property
    def thumb_tip(self):
        return self.landmarks[4]

    @property
    def index_tip(self):
        return self.landmarks[8]

    @property
    def wrist(self):
        return self.landmarks[0]


class HandTracker:
    def __init__(self, model_path=None, max_hands=2, det_conf=0.6, track_conf=0.5,
                 pinch_on=0.25, pinch_off=0.40):
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=track_conf,
            min_tracking_confidence=track_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._filters, self._last_seen, self._pinch = {}, {}, {}
        self._last_ts = 0
        self.pinch_on, self.pinch_off = pinch_on, pinch_off

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        res = self._lm.detect_for_video(mp_img, ts)

        now = time.monotonic()
        used, hands = set(), []
        for lm, hd in zip(res.hand_landmarks, res.handedness):
            label = hd[0].category_name
            if label in used:                      # both reported as same side
                label = "Left" if label == "Right" else "Right"
            used.add(label)

            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
            f = self._filters.setdefault(label, OneEuroFilter(min_cutoff=1.5, beta=0.02))
            if now - self._last_seen.get(label, 0) > 0.3:
                f.reset()
                self._pinch[label] = False
            self._last_seen[label] = now
            pts = f(pts, now)

            palm = max(np.linalg.norm(pts[0] - pts[9]), 1.0)
            ratio = float(np.linalg.norm(pts[4] - pts[8]) / palm)
            was = self._pinch.get(label, False)
            pinching = ratio < (self.pinch_off if was else self.pinch_on)   # hysteresis
            self._pinch[label] = pinching
            hands.append(Hand(label, pts, ratio, pinching))

        hands.sort(key=lambda hd: hd.wrist[0])
        return hands

    def close(self):
        self._lm.close()
