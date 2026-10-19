"""Non-camera regression checks for the image-processing pipeline."""
import time
import threading
import unittest

import numpy as np

import inference
from inference import StyleEngine
from perspective import order_quad, quad_valid, unwarp, warp_into
from smoothing import OneEuroFilter
from styles import STYLES
from ui_overlay import draw_thumbnail


class CorePipelineTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:240, :320]
        self.frame = np.dstack(((x % 256), (y % 256), ((x + y) % 256))).astype(np.uint8)
        self.quad = np.float32([[40, 30], [280, 45], [260, 205], [55, 190]])

    def test_quad_ordering_and_validation(self):
        ordered = order_quad(self.quad[[2, 0, 3, 1]])
        self.assertTrue(quad_valid(ordered, self.frame.shape))
        self.assertFalse(quad_valid(np.zeros((4, 2), np.float32), self.frame.shape))
        self.assertFalse(quad_valid(np.float32([[-1, 0], [50, 0], [50, 50], [0, 50]]), self.frame.shape))

    def test_perspective_round_trip_preserves_canvas(self):
        capture = unwarp(self.frame, self.quad)
        self.assertGreaterEqual(max(capture.shape[:2]), 500)
        composited = warp_into(self.frame, capture, self.quad)
        self.assertEqual(composited.shape, self.frame.shape)
        self.assertEqual(composited.dtype, np.uint8)

    def test_all_local_styles_return_valid_images(self):
        for style in STYLES:
            with self.subTest(style=style.name):
                output = style.fallback(self.frame)
                self.assertEqual(output.shape, self.frame.shape)
                self.assertEqual(output.dtype, np.uint8)

    def test_style_engine_clears_dropped_capture(self):
        engine = StyleEngine()
        try:
            engine.request(1, self.frame, 0)
            deadline = time.monotonic() + 3
            while engine.get(1, 0) is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(engine.get(1, 0))
            engine.drop_capture(1)
            self.assertIsNone(engine.get(1, 0))
        finally:
            engine.shutdown()

    def test_inflight_style_result_is_discarded_after_reset(self):
        """A late worker result must not bring a reset capture back to the UI."""
        started, release = threading.Event(), threading.Event()
        original = inference.STYLES[0]

        def slow_style(image):
            started.set()
            release.wait(timeout=2)
            return image

        inference.STYLES[0] = type(original)("Slow test", "", 0.0, slow_style)
        engine = StyleEngine()
        try:
            engine.request(99, self.frame, 0)
            self.assertTrue(started.wait(timeout=1))
            engine.drop_capture(99)
            release.set()
            deadline = time.monotonic() + 2
            while engine.is_pending(99, 0) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(engine.is_pending(99, 0))
            self.assertIsNone(engine.get(99, 0))
        finally:
            release.set()
            engine.shutdown()
            inference.STYLES[0] = original

    def test_thumbnail_handles_tiny_canvas(self):
        canvas = np.zeros((20, 20, 3), np.uint8)
        draw_thumbnail(canvas, self.frame)
        self.assertEqual(canvas.shape, (20, 20, 3))

    def test_filter_remains_finite(self):
        filt = OneEuroFilter()
        values = [filt(np.array([float(i), 10.0]), i / 30) for i in range(10)]
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
