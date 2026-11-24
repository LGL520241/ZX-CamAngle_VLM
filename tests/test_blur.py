import unittest

import cv2
import numpy as np

from cam_angle.blur import analyze_blur


def scene(width=640, height=480):
    """含有多处真实几何结构，而不是随机像素的可重复测试场景。"""
    img = np.full((height, width, 3), 190, np.uint8)
    rng = np.random.default_rng(49)
    for y in range(10, height - 20, 45):
        for x in range(10, width - 20, 45):
            color = tuple(int(c) for c in rng.integers(25, 230, 3))
            cv2.rectangle(img, (x, y), (x + 27, y + 29), color, -1)
            cv2.line(img, (x, y + 12), (x + 27, y + 12), (15, 15, 15), 2)
    return img


class BlurTests(unittest.TestCase):
    def test_sharp_color_and_grayscale(self):
        img = scene()
        self.assertEqual(analyze_blur(img, {})["status"], "pass")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(analyze_blur(gray, {})["status"], "pass")

    def test_defocus_with_sharp_timestamp(self):
        img = cv2.GaussianBlur(scene(), (31, 31), 7)
        cv2.putText(img, "2026-11-20 12:30:00", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        result = analyze_blur(img, {})
        self.assertEqual(result["status"], "fail", result)
        self.assertGreater(result["metrics"]["blur_affected_ratio"], .7)

    def test_motion_blur(self):
        kernel = np.zeros((25, 25), np.float32)
        kernel[12, :] = 1 / 25
        result = analyze_blur(cv2.filter2D(scene(), -1, kernel), {})
        self.assertIn(result["status"], ("fail", "warn"), result)

    def test_partial_blur(self):
        img = scene()
        blurred = cv2.GaussianBlur(img, (31, 31), 7)
        img[:, :160] = blurred[:, :160]
        self.assertEqual(analyze_blur(img, {})["status"], "warn")

    def test_plain_surface_and_gradient_are_not_defocus(self):
        for img in (np.full((480, 640, 3), 128, np.uint8),
                    np.repeat(np.tile(np.linspace(20, 220, 640).astype(np.uint8), (480, 1))[:, :, None], 3, axis=2)):
            self.assertEqual(analyze_blur(img, {})["status"], "unknown")

    def test_noise_is_not_sharp(self):
        rng = np.random.default_rng(42)
        img = np.clip(rng.normal(128, 30, (480, 640, 3)), 0, 255).astype(np.uint8)
        self.assertEqual(analyze_blur(img, {})["status"], "warn")

    def test_occluded_part_does_not_count_as_blurry(self):
        img = scene()
        img[:, :320] = 80
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[:, :320] = 255
        result = analyze_blur(img, {}, exclusion_mask=mask)
        self.assertEqual(result["status"], "pass", result)

    def test_all_excluded_and_tiny_images(self):
        img = scene()
        self.assertEqual(analyze_blur(img, {}, exclusion_mask=np.ones(img.shape[:2], np.uint8))["status"], "unknown")
        self.assertEqual(analyze_blur(img[:32, :32], {})["status"], "unknown")

    def test_large_resolution_normalization(self):
        img = cv2.resize(scene(), (1920, 1440), interpolation=cv2.INTER_NEAREST)
        result = analyze_blur(img, {})
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["metrics"]["blur_analysis_width"], 640)

    def test_bad_mask_rejected(self):
        with self.assertRaises(ValueError):
            analyze_blur(scene(), {}, exclusion_mask=np.zeros((10, 10), np.uint8))


if __name__ == "__main__":
    unittest.main()
