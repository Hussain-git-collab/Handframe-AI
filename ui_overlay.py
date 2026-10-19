"""HUD and drawing helpers."""
import cv2
import numpy as np

from hand_tracker import HAND_CONNECTIONS

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (255, 200, 60)


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_hand(img, hand):
    pts = hand.landmarks.astype(int)
    col = (80, 255, 120) if hand.pinching else (230, 230, 230)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (120, 120, 120), 1, cv2.LINE_AA)
    for i in (4, 8):
        cv2.circle(img, tuple(pts[i]), 8, col, -1, cv2.LINE_AA)
    if hand.pinching:
        cv2.line(img, tuple(pts[4]), tuple(pts[8]), col, 3, cv2.LINE_AA)


def draw_quad(img, quad, color=ACCENT):
    q = np.int32(quad)
    cv2.polylines(img, [q], True, color, 2, cv2.LINE_AA)
    for p in q:
        cv2.circle(img, tuple(p), 6, color, -1, cv2.LINE_AA)


def flash(img, amount):
    if amount > 0:
        return cv2.addWeighted(img, 1 - amount, np.full_like(img, 255), amount, 0)
    return img


def draw_thumbnail(img, panel, margin=16, width=220):
    h, w = panel.shape[:2]
    H, W = img.shape[:2]
    margin = min(margin, max((min(H, W) - 2) // 4, 0))
    width = min(width, max(W - 2 * margin, 1))
    th = min(int(width * h / w), max(H - 2 * margin - 30, 1))
    width = max(int(th * w / h), 1)
    x0, y0 = max(W - width - margin, 0), max(H - th - margin - 30, 0)
    img[y0:y0 + th, x0:x0 + width] = cv2.resize(panel, (width, th))
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + th), (255, 255, 255), 2)


def draw_hud(img, fps, style_name, idx, total, status, hint, busy=False):
    H, W = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (W, 38), (0, 0, 0), -1)
    cv2.rectangle(bar, (0, H - 34), (W, H), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
    text(img, f"HandFrame AI   {fps:4.0f} fps", (12, 26), 0.6, ACCENT, 1)
    label = f"[ <  {style_name}  ({idx + 1}/{total})  > ]" + ("   generating..." if busy else "")
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    text(img, label, ((W - tw) // 2, H - 12), 0.6, (255, 255, 255), 1)
    text(img, status, (W - 260, 26), 0.5, (200, 200, 200), 1)
    text(img, hint, (12, H - 12), 0.5, (180, 220, 255), 1)
