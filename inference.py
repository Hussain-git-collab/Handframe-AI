"""Asynchronous style engine: instant OpenCV styles, optional FLUX.1 img2img."""
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from styles import STYLES


class StyleEngine:
    def __init__(self, use_flux=False, model_path=None, size=512, steps=8):
        self.use_flux = use_flux
        self.model_path = model_path or "black-forest-labs/FLUX.1-schnell"
        self.size, self.steps = size, steps
        self.status = "OpenCV styles" if not use_flux else "FLUX (loads on first capture)"
        self._pool = ThreadPoolExecutor(max_workers=1)   # one job at a time (GPU safe)
        self._cache, self._pending, self._discarded = {}, set(), set()
        self._lock = threading.Lock()
        self._pipe = None

    # --- public API (non-blocking) ---
    def get(self, capture_id, style_idx):
        with self._lock:
            return self._cache.get((capture_id, style_idx))

    def is_pending(self, capture_id, style_idx):
        with self._lock:
            return (capture_id, style_idx) in self._pending

    def request(self, capture_id, image, style_idx):
        key = (capture_id, style_idx)
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            self._pending.add(key)
        self._pool.submit(self._run, key, image.copy())

    def drop_capture(self, capture_id):
        """Forget a capture, including results from jobs that finish later."""
        with self._lock:
            for k in [k for k in self._cache if k[0] == capture_id]:
                del self._cache[k]
            self._discarded.add(capture_id)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    # --- internals ---
    def _run(self, key, image):
        _, style_idx = key
        style = STYLES[style_idx]
        try:
            out = self._flux(image, style) if self.use_flux else None
            if out is None:
                out = style.fallback(image)
        except Exception as e:                      # never kill the app from a worker
            print(f"[HandFrame] styling error: {e}")
            out = image
        with self._lock:
            # A FLUX job may complete after the user has captured a new frame.
            # Never let that stale output re-enter the cache.
            if key[0] not in self._discarded:
                self._cache[key] = out
            self._pending.discard(key)

    def _load_flux(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import FluxImg2ImgPipeline
            self.status = "Loading FLUX model..."
            cuda = torch.cuda.is_available()
            dtype = torch.bfloat16 if (cuda or torch.backends.mps.is_available()) else torch.float32
            pipe = FluxImg2ImgPipeline.from_pretrained(self.model_path, torch_dtype=dtype)
            if cuda:
                pipe.enable_model_cpu_offload()
            elif torch.backends.mps.is_available():
                pipe.to("mps")
            self._pipe, self._torch = pipe, torch
            self.status = "FLUX ready"
        except Exception as e:
            print(f"[HandFrame] FLUX unavailable ({e}); using OpenCV styles.")
            self.use_flux, self.status = False, "FLUX failed -> OpenCV styles"
        return self._pipe

    def _flux(self, image, style):
        pipe = self._load_flux()
        if pipe is None:
            return None
        from PIL import Image
        h, w = image.shape[:2]
        k = self.size / max(h, w)
        nw, nh = max(64, int(w * k) // 16 * 16), max(64, int(h * k) // 16 * 16)   # multiples of 16
        pil = Image.fromarray(cv2.cvtColor(cv2.resize(image, (nw, nh)), cv2.COLOR_BGR2RGB))
        self.status = f"FLUX: {style.name}"
        res = pipe(prompt=style.prompt, image=pil, strength=style.strength,
                   num_inference_steps=self.steps, guidance_scale=0.0,
                   height=nh, width=nw).images[0]
        self.status = "FLUX ready"
        return cv2.resize(cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR), (w, h))
