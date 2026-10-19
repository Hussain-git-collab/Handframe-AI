"""Quad ordering, perspective unwarp (capture) and warp-back (floating panel)."""
import cv2
import numpy as np


def order_quad(pts):
    """Return 4 points ordered TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("A quadrilateral needs exactly four 2D points")

    # Sorting by sum/difference is tempting, but it duplicates corners for
    # rotated diamonds.  Sorting around the centroid works for every convex
    # four-point frame and gives a stable order before perspective mapping.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    top_left = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -top_left, axis=0).astype(np.float32)


def quad_size(q):
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


def quad_valid(q, frame_shape, min_side=70, min_area_frac=0.02, max_aspect=4.0):
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False
    H, W = frame_shape[:2]
    if np.unique(np.round(q, 1), axis=0).shape[0] != 4:
        return False
    if (q[:, 0].min() < 0 or q[:, 0].max() >= W or
            q[:, 1].min() < 0 or q[:, 1].max() >= H):
        return False
    w, h = quad_size(q)
    if min(w, h) < min_side or max(w, h) / max(min(w, h), 1) > max_aspect:
        return False
    contour = q.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False
    return cv2.contourArea(contour) >= min_area_frac * W * H


def unwarp(frame, quad, max_side=512):
    """Rectify the region inside `quad` into an upright image."""
    w, h = quad_size(quad)
    scale = max_side / max(w, h)
    ow, oh = max(32, int(w * scale)), max(32, int(h * scale))
    dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(frame, M, (ow, oh), flags=cv2.INTER_CUBIC)


def center_crop_to_aspect(img, aspect):
    h, w = img.shape[:2]
    if w / h > aspect:
        nw = int(h * aspect)
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(w / aspect)
    y0 = (h - nh) // 2
    return img[y0:y0 + nh, :]


def warp_into(frame, panel, quad, shadow=True, border=(255, 255, 255)):
    """Paste `panel` onto `frame` inside `quad` with perspective, shadow and border."""
    H, W = frame.shape[:2]
    qw, qh = quad_size(quad)
    panel = center_crop_to_aspect(panel, qw / max(qh, 1))
    ph, pw = panel.shape[:2]
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(panel, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H))

    out = frame.copy()
    if shadow:
        sh = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(sh, (np.float32(quad) + [10, 14]).astype(np.int32), 255)
        sh = cv2.GaussianBlur(sh, (0, 0), 12).astype(np.float32) / 255.0
        out = (out * (1 - 0.45 * sh[..., None])).astype(np.uint8)

    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = (out * (1 - m) + warped * m).astype(np.uint8)
    if border is not None:
        cv2.polylines(out, [np.int32(quad)], True, border, 2, cv2.LINE_AA)
    return out
