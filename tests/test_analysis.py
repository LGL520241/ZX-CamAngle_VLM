import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from cam_angle.analysis import AnalysisEngine
from cam_angle.config import Config
from cam_angle.database import Database
from cam_angle.pipeline import Pipeline
from test_blur import scene


def sharp_scene():
    image = scene()
    background = np.all(image == 190, axis=2)
    image[background] = (50, 140, 210)
    return image


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.cfg = Config(data_dir=self.root, db_path=self.root / 'test.db',
                          demo_mode=True, hik_client_id='', hik_client_secret='',
                          region_filter='', sample_size=0, vlm_enabled=False,
                          artifact_root=self.root / 'artifacts',
                          module_switches={
                              'occlusion': True,
                              'blur': True,
                              'qwen_semantic': False,
                          })
        self.cfg.ensure_dirs()
        self.cfg.thresholds["occlusion_area_ratio"] = 0.25
        self.db = Database(self.cfg.db_path)
        self.camera_id = self.db.upsert_camera('TEST', 1, meta={'demo': True, 'kind': 'normal'})
        self.camera = self.db.get_camera(self.camera_id)
        self.engine = AnalysisEngine(self.cfg, self.db)

    def tearDown(self):
        self.db._conn.close()
        self.folder.cleanup()

    def analyze(self, image, name='image.png'):
        path = self.root / name
        ok, data = cv2.imencode('.png', image)
        self.assertTrue(ok)
        path.write_bytes(data.tobytes())
        return self.engine.analyze(path, self.camera), path

    def test_normal_image_passes_without_reading_baseline(self):
        with patch.object(self.db, 'get_latest_baseline', side_effect=AssertionError('must not use baseline')):
            result, path = self.analyze(sharp_scene(), '中文摄像头.png')
        self.assertEqual(result['decision'], 'pass', result)
        self.assertEqual(result['module_results']['baseline']['status'], 'skipped')
        self.assertIsNone(result['baseline_id'])
        self.assertTrue(result['module_results']['occlusion']['overlay_available'])
        self.assertTrue(path.with_suffix('.occlusion.png').is_file())

    def test_large_obstruction_fails_with_clear_remainder(self):
        image = sharp_scene()
        image[:, :220] = 12
        result, _ = self.analyze(image)
        self.assertEqual(result['decision'], 'fail', result)
        self.assertEqual(result['module_results']['occlusion']['status'], 'fail')
        self.assertIn(result['module_results']['blur']['status'], ('pass', 'warn'))

    def test_blurred_scene_fails_despite_sharp_timestamp(self):
        image = cv2.GaussianBlur(sharp_scene(), (31, 31), 7)
        cv2.putText(image, '2026-11-20 12:00:00', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        result, _ = self.analyze(image)
        self.assertEqual(result['decision'], 'fail', result)
        self.assertEqual(result['module_results']['blur']['status'], 'fail', result)

    def test_combined_obstruction_and_blur_both_reported(self):
        image = cv2.GaussianBlur(sharp_scene(), (31, 31), 7)
        image[:, :180] = 15
        result, _ = self.analyze(image)
        self.assertEqual(result['decision'], 'fail', result)
        self.assertEqual(result['module_results']['occlusion']['status'], 'fail')
        self.assertIn(result['module_results']['blur']['status'], ('fail', 'warn'))

    def test_overlay_write_failure_preserves_analysis(self):
        with patch('cam_angle.analysis.save_occlusion_overlay', return_value=False):
            result, _ = self.analyze(sharp_scene())
        self.assertEqual(result['decision'], 'pass', result)
        self.assertFalse(result['module_results']['occlusion']['overlay_available'])

    def test_pipeline_persists_modules_and_evidence_without_baseline(self):
        pipeline = Pipeline(self.cfg, self.db)
        with patch('cam_angle.pipeline.render_demo_image', return_value=sharp_scene()):
            result = asyncio.run(pipeline.run_cycle())
        self.assertEqual(result['status'], 'finished')
        capture = self.db.list_captures(camera_id=self.camera_id)[0]
        self.assertEqual(capture['decision'], 'pass', capture)
        self.assertIn('blur', capture['module_results'])
        self.assertIn('occlusion', capture['module_results'])
        self.assertIsNone(capture['baseline_id'])
        self.assertIsNotNone(capture['batch_id'])
        batch = self.db.get_batch(capture['batch_id'])
        self.assertEqual(batch['state'], 'finished')
        self.assertEqual(batch['processed'], 1)
        self.assertTrue(Path(capture['path']).with_suffix('.occlusion.png').is_file())

    def test_module_switch_skips_cv_modules_without_failing_capture(self):
        path = self.root / 'switches.jpg'
        ok, data = cv2.imencode('.jpg', sharp_scene())
        self.assertTrue(ok)
        path.write_bytes(data.tobytes())
        result = self.engine.analyze(
            path,
            self.camera,
            {'occlusion': False, 'blur': False, 'qwen_semantic': False},
        )
        self.assertEqual(result['module_results']['occlusion']['status'], 'skipped')
        self.assertEqual(result['module_results']['blur']['status'], 'skipped')
        self.assertNotIn('occlusion_ratio', result['metrics'])
        self.assertEqual(result['module_results']['quality']['status'], 'pass')
        self.assertFalse(path.with_suffix('.occlusion.png').exists())


if __name__ == '__main__':
    unittest.main()
