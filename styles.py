"""16 style presets. Each has a FLUX prompt and an instant OpenCV fallback."""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


# ---------- helpers ----------
def _gray3(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _edges(img, block=9, c=7):
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)


def _quantize(img, levels):
    step = max(256 // levels, 1)
    return ((img // step) * step + step // 2).astype(np.uint8)


def _vignette(img, strength=0.6):
    h, w = img.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.55)
    ky = cv2.getGaussianKernel(h, h * 0.55)
    k = ky @ kx.T
    k = (k / k.max()) ** strength
    return (img * k[..., None]).astype(np.uint8)


def _glow(img, ksize=0, sigma=9, weight=0.8):
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return cv2.addWeighted(img, 1.0, blur, weight, 0)


def _hsv_boost(img, s=1.4, v=1.05):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------- fallback stylizers (BGR uint8 -> BGR uint8) ----------
def pencil(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.GaussianBlur(255 - g, (21, 21), 0)
    return _gray3(cv2.divide(g, 255 - inv, scale=256))


def color_pencil(img):
    return cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[1]


def cartoon(img):
    s = img
    for _ in range(3):
        s = cv2.bilateralFilter(s, 9, 40, 7)
    return cv2.bitwise_and(_quantize(s, 8), _gray3(_edges(img)))


def anime(img):
    s = cv2.pyrMeanShiftFiltering(img, 8, 28)
    s = _hsv_boost(_quantize(s, 6), 1.25, 1.08)
    lines = cv2.erode(_edges(img, 7, 6), np.ones((2, 2), np.uint8))
    return cv2.bitwise_and(s, _gray3(lines))


def oil(img):
    return cv2.stylization(img, sigma_s=70, sigma_r=0.55)


def watercolor(img):
    s = cv2.stylization(img, sigma_s=40, sigma_r=0.35)
    paper = cv2.GaussianBlur(np.random.default_rng(1).normal(235, 8, img.shape[:2]).astype(np.float32), (0, 0), 1.2)
    paper = np.clip(paper, 0, 255).astype(np.uint8)
    return cv2.addWeighted(s, 0.85, _gray3(paper), 0.15, 0)


def pop_art(img):
    return _hsv_boost(_quantize(cv2.bilateralFilter(img, 9, 60, 9), 4), 1.8, 1.1)


def neon(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 50, 140), np.ones((2, 2), np.uint8))
    hue = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_COOL)
    out = cv2.bitwise_and(hue, _gray3(e))
    return _glow(np.clip(out * 1.4, 0, 255).astype(np.uint8), sigma=7, weight=1.2)


def halftone(img, cell=9):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out = np.full((h, w, 3), 250, np.uint8)
    for y in range(small.shape[0]):
        for x in range(small.shape[1]):
            r = int((1 - gray[y, x] / 255.0) * cell * 0.75) + 1
            c = tuple(int(v) for v in _hsv_boost(small[y:y + 1, x:x + 1], 1.6)[0, 0])
            cv2.circle(out, (x * cell + cell // 2, y * cell + cell // 2), r, c, -1, cv2.LINE_AA)
    return cv2.resize(out, (w, h))


def cyberpunk(img):
    b, g, r = cv2.split(cv2.convertScaleAbs(img, alpha=1.3, beta=-25))
    out = cv2.merge([cv2.convertScaleAbs(b, alpha=1.3), cv2.convertScaleAbs(g, alpha=0.6),
                     cv2.convertScaleAbs(r, alpha=1.15)])
    e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 170)
    edge_col = np.zeros_like(out)
    edge_col[..., 0], edge_col[..., 1], edge_col[..., 2] = 255, 230, 40   # cyan lines
    out = np.where(e[..., None] > 0, edge_col, out)
    return _glow(out.astype(np.uint8), sigma=8, weight=0.6)


def vaporwave(img):
    h, w = img.shape[:2]
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top, bot = np.float32([180, 60, 255]), np.float32([230, 220, 40])     # BGR pink -> cyan
    grad = np.broadcast_to(top * (1 - t) + bot * t, (h, w, 3))
    g = _gray3(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    return np.clip(0.55 * grad + 0.55 * g * 0.9, 0, 255).astype(np.uint8)


def vintage(img):
    k = np.float32([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    s = np.clip(cv2.transform(img, k), 0, 255).astype(np.uint8)
    noise = np.random.default_rng(2).normal(0, 12, s.shape).astype(np.int16)
    return _vignette(np.clip(s.astype(np.int16) + noise, 0, 255).astype(np.uint8), 0.8)


def noir(img):
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    g = cv2.convertScaleAbs(g, alpha=1.25, beta=-30)
    return _vignette(_gray3(g), 0.9)


def pixel(img):
    h, w = img.shape[:2]
    s = cv2.resize(img, (max(w // 12, 8), max(h // 12, 8)), interpolation=cv2.INTER_AREA)
    return cv2.resize(_quantize(s, 6), (w, h), interpolation=cv2.INTER_NEAREST)


def mosaic(img, cell=18):
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
    out = cv2.resize(_hsv_boost(s, 1.3), (w, h), interpolation=cv2.INTER_NEAREST)
    out[::cell, :] = 30
    out[:, ::cell] = 30
    return out


def thermal(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    return cv2.applyColorMap(cv2.equalizeHist(g), cv2.COLORMAP_INFERNO)


def ascii_terminal(img):
    """Render luminance as high-contrast, projection-friendly ASCII art."""
    h, w = img.shape[:2]
    # Deliberately use large, non-antialiased glyphs: thin anti-aliased text
    # looks soft after the perspective warp and is hard to read on a projector.
    cell_w, cell_h = 10, 16
    cols, rows = max(w // cell_w, 1), max(h // cell_h, 1)
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (cols, rows),
                      interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    charset = "@#S%?*+;:,. "  # dark pixels receive dense characters
    indices = np.clip(gray.astype(np.int16) * len(charset) // 256, 0, len(charset) - 1)
    out = np.zeros_like(img)
    scale = 1.0
    for row in range(rows):
        y = min((row + 1) * cell_h - 4, h - 1)
        for col in range(cols):
            char = charset[indices[row, col]]
            if char != " ":
                cv2.putText(out, char, (col * cell_w, y), cv2.FONT_HERSHEY_PLAIN,
                            scale, (80, 255, 130), 1, cv2.LINE_8)
    return out


# ---------- presets ----------
@dataclass(frozen=True)
class Style:
    name: str
    prompt: str
    strength: float
    fallback: Callable[[np.ndarray], np.ndarray]


_KEEP = ", portrait of the same person, preserve face and pose, high quality"

STYLES = [
    Style("Pencil Sketch", "detailed graphite pencil sketch on white paper, fine cross-hatching" + _KEEP, 0.75, pencil),
    Style("Color Pencil", "colored pencil illustration, soft strokes, paper texture" + _KEEP, 0.7, color_pencil),
    Style("Cartoon", "bold western cartoon, flat colors, thick black outlines" + _KEEP, 0.7, cartoon),
    Style("Anime", "studio anime portrait, cel shading, clean line art, vibrant" + _KEEP, 0.75, anime),
    Style("Oil Painting", "classical oil painting, visible brush strokes, rich impasto" + _KEEP, 0.7, oil),
    Style("Watercolor", "delicate watercolor painting, soft washes, paper grain" + _KEEP, 0.72, watercolor),
    Style("Pop Art", "andy warhol pop art, saturated colors, high contrast" + _KEEP, 0.75, pop_art),
    Style("Neon Glow", "glowing neon line art on black background, vivid light trails" + _KEEP, 0.8, neon),
    Style("Comic Halftone", "vintage comic book print, halftone dots, ink outlines" + _KEEP, 0.75, halftone),
    Style("Cyberpunk", "cyberpunk neon city portrait, cyan and magenta lighting" + _KEEP, 0.75, cyberpunk),
    Style("Vaporwave", "vaporwave aesthetic, pink and cyan gradient, retro 80s" + _KEEP, 0.75, vaporwave),
    Style("Vintage Film", "1970s vintage film photograph, faded colors, film grain" + _KEEP, 0.6, vintage),
    Style("Film Noir", "black and white film noir, dramatic shadows, high contrast" + _KEEP, 0.65, noir),
    Style("Pixel Art", "16-bit pixel art sprite portrait, limited palette" + _KEEP, 0.8, pixel),
    Style("Mosaic", "ancient roman tile mosaic, small colored tiles, grout lines" + _KEEP, 0.8, mosaic),
    Style("Thermal", "thermal infrared camera image, inferno color palette" + _KEEP, 0.8, thermal),
    Style("ASCII Terminal", "green terminal ASCII art made from text characters on a black screen" + _KEEP, 0.75, ascii_terminal),
]
