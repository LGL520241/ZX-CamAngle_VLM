import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from cam_angle.config import Config
from cam_angle.occlusion import analyze_occlusion, save_occlusion_overlay


def colorful_scene():
    rng = np.random.default_rng(2026)
    img = np.full((400, 600, 3), (80, 160, 230), np.uint8)
    for _ in range(180):
        x, y = int(rng.integers(0, 580)), int(rng.integers(0, 380))
        color = (int(rng.integers(40, 100)), int(rng.integers(110, 180)), int(rng.integers(190, 250)))
        cv2.rectangle(img, (x, y), (x + 18, y + 18), color, -1)
    return img


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.t = Config().thresholds
        self.t["occlusion_area_ratio"] = 0.25

    def analyze(self, img):
        return analyze_occlusion(img, self.t)

    def test_black_and_gray_over_quarter(self):
        for color in (0, 22, 100, 190):
            with self.subTest(color=color):
                img = colorful_scene()
                img[:, :180] = color
                result, mask = self.analyze(img)
                self.assertEqual(result['status'], 'fail')
                self.assertAlmostEqual(result['metrics']['occlusion_ratio'], .3, delta=.012)
                self.assertEqual(mask.shape, img.shape[:2])

    def test_threshold_is_strictly_greater_than_25_percent(self):
        for rows, fail in ((80, False), (100, False), (108, True)):
            with self.subTest(rows=rows):
                img = colorful_scene()
                img[:rows] = 0
                result, _ = self.analyze(img)
                self.assertEqual(result['status'] == 'fail', fail)

    def test_multiple_disconnected_regions_use_total_pixel_area(self):
        img = colorful_scene()
        img[:100, :400] = 20
        img[300:, 200:] = 110
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['metrics']['region_count'], 2)
        self.assertAlmostEqual(result['metrics']['occlusion_ratio'], 1/3, delta=.015)

    def test_bounding_box_does_not_count_as_covered_area(self):
        img = colorful_scene()
        img[20:380, 20:32] = 0
        img[20:32, 20:580] = 0
        img[368:380, 20:580] = 0
        result, _ = self.analyze(img)
        self.assertNotEqual(result['status'], 'fail')
        self.assertLess(result['metrics']['occlusion_ratio'], .12)

    def test_textured_gray_scene_is_not_a_gray_occluder(self):
        rng = np.random.default_rng(5)
        gray = rng.integers(40, 210, (400, 600), dtype=np.uint8)
        result, _ = self.analyze(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        self.assertEqual(result['status'], 'pass')
        self.assertLess(result['metrics']['occlusion_ratio'], .05)

    def test_saturated_object_is_outside_black_gray_rule(self):
        img = colorful_scene()
        img[:, :300] = (20, 20, 180)
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'pass')

    def test_gray_obstruction_with_sensor_noise(self):
        img = colorful_scene()
        noise = np.random.default_rng(7).normal(0, 2, (400, 200, 1))
        img[:, :200] = np.clip(120 + noise, 0, 255).astype(np.uint8)
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'fail')

    def test_small_scattered_spots_are_not_aggregated(self):
        img = colorful_scene()
        for y in range(0, 400, 10):
            for x in range(0, 600, 10):
                img[y:y+4, x:x+4] = 0
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'pass')

    def test_resolutions_and_jpeg_preserve_large_obstruction(self):
        img = colorful_scene()
        img[:, :200] = 110
        for size in ((300, 200), (1200, 800), (2400, 1600)):
            with self.subTest(size=size):
                scaled = cv2.resize(img, size)
                ok, encoded = cv2.imencode('.jpg', scaled, [cv2.IMWRITE_JPEG_QUALITY, 75])
                self.assertTrue(ok)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                result, mask = self.analyze(decoded)
                self.assertEqual(result['status'], 'fail')
                self.assertAlmostEqual(result['metrics']['occlusion_ratio'], 1/3, delta=.025)
                self.assertEqual(mask.shape, decoded.shape[:2])

    def test_overlay_is_transparent_and_unicode_safe(self):
        img = colorful_scene()
        img[:, :200] = 0
        _, mask = self.analyze(img)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '摄像头.jpg'
            self.assertTrue(save_occlusion_overlay(mask, path))
            png = cv2.imdecode(np.frombuffer(path.with_suffix('.occlusion.png').read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertEqual(png.shape, (400, 600, 4))
            self.assertEqual(int(png[200, 10, 3]), 110)
            self.assertEqual(int(png[200, 500, 3]), 0)

    def test_small_picture_requires_review(self):
        result, _ = self.analyze(np.zeros((32, 32, 3), np.uint8))
        self.assertEqual(result['status'], 'unknown')

    def test_opposite_letterbox_bars_are_ignored(self):
        img = colorful_scene()
        img[:50] = 0
        img[-50:] = 0
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'pass', result)
        self.assertAlmostEqual(result['metrics']['letterbox_ratio'], .25, delta=.01)
        self.assertLess(result['metrics']['occlusion_ratio'], .05)

    def test_single_side_black_band_is_still_occlusion(self):
        img = colorful_scene()
        img[:105] = 0
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(result['metrics']['letterbox_ratio'], 0)
        self.assertAlmostEqual(result['metrics']['occlusion_ratio'], .2625, delta=.012)

    def test_thick_black_blocks_are_not_treated_as_letterbox(self):
        img = colorful_scene()
        img[:100] = 0
        img[-100:] = 0
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(result['metrics']['letterbox_ratio'], 0)

    def test_pillarbox_bars_are_ignored(self):
        img = colorful_scene()
        img[:, :50] = 0
        img[:, -50:] = 0
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'pass', result)
        self.assertLess(result['metrics']['occlusion_ratio'], .05)

    def test_threshold_setting_changes_decision(self):
        img = colorful_scene()
        img[:, :180] = 0
        self.t['occlusion_area_ratio'] = .4
        result, _ = self.analyze(img)
        self.assertEqual(result['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
