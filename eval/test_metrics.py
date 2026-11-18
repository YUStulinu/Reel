"""python -m unittest discover eval"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402


def write_y4m(path, frames, w, h):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 C420jpeg\n".encode())
        for y, u, v in frames:
            f.write(b"FRAME\n" + y.tobytes() + u.tobytes() + v.tobytes())


class Metrics(unittest.TestCase):
    def test_psnr_of_a_known_error(self):
        # every sample off by 4: MSE 16 -> 10 log10(255^2/16)
        self.assertAlmostEqual(metrics.psnr_from_mse(16), 36.0896, places=3)
        self.assertEqual(metrics.psnr_from_mse(0), float("inf"))

    def test_ssim(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (64, 64)).astype(np.uint8)
        self.assertAlmostEqual(metrics.ssim(a, a), 1.0, places=9)
        noisy = np.clip(a.astype(int) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
        s = metrics.ssim(a, noisy)
        self.assertTrue(0.3 < s < 0.99, s)
        # a constant offset barely changes the structure term
        shifted = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
        self.assertGreater(metrics.ssim(a, shifted), s)

    def test_compare_reads_sequences(self):
        w, h = 32, 16
        y = np.full((h, w), 100, np.uint8)
        c = np.full((h // 2, w // 2), 128, np.uint8)
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.y4m"), os.path.join(d, "b.y4m")
            write_y4m(a, [(y, c, c)] * 3, w, h)
            write_y4m(b, [(y + 2, c, c)] * 3, w, h)
            r = metrics.compare(a, b)
            self.assertEqual(r["frames"], 3)
            self.assertAlmostEqual(r["psnr_y"], metrics.psnr_from_mse(4))
            self.assertEqual(r["psnr_u"], float("inf"))
            self.assertEqual(metrics.y4m_info(a), (w, h, 25.0))

    def test_bd_rate(self):
        rates = np.array([100.0, 200, 400, 800])
        q = np.array([30.0, 33, 36, 39])
        self.assertAlmostEqual(metrics.bd_rate(rates, q, rates, q), 0.0, places=6)
        # the same quality for 20% fewer bits
        self.assertAlmostEqual(metrics.bd_rate(rates, q, rates * 0.8, q), -20.0, places=4)
        # +1 dB everywhere on a 3 dB/octave curve is 1/3 octave of rate
        self.assertAlmostEqual(metrics.bd_rate(rates, q, rates, q + 1), (2 ** (-1 / 3) - 1) * 100, places=3)
        self.assertAlmostEqual(metrics.bd_quality(rates, q, rates, q + 1), 1.0, places=6)
        self.assertIsNone(metrics.bd_rate(rates, q, rates, q + 20))


if __name__ == "__main__":
    unittest.main()
