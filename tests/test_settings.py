import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from cam_angle.config import (
    Config,
    restore_module_switches,
    restore_thresholds,
    validate_module_switches,
    validate_thresholds,
)
from cam_angle.database import Database


class AsgiTestClient:
    """直接请求 ASGI 应用，兼容仓库使用的不同 Starlette 版本。"""
    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


class SettingsValidationTests(unittest.TestCase):
    def setUp(self):
        self.defaults = Config().thresholds

    def test_rejects_non_numeric_unknown_and_nonfinite_values(self):
        for key, value in (
            ("occlusion_area_ratio", True),
            ("occlusion_area_ratio", "0.25"),
            ("occlusion_area_ratio", float("nan")),
            ("occlusion_area_ratio", float("inf")),
            ("occlusion_area_ratio", -0.1),
            ("occlusion_area_ratio", 25),
            ("occlusion_area_ratio", 10 ** 400),
            ("occlusion_dark_value", 256),
            ("baseline_min_good", 8.5),
            ("blur_min_tiles", 4.5),
            ("blur_min_tiles", 17),
            ("blur_region_ratio", 0),
            ("blur_region_ratio", 50),
            ("blur_var", 0),
            ("unknown_threshold", 1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_thresholds(self.defaults, {key: value})
        self.assertEqual(self.defaults["occlusion_area_ratio"], 0.35)

    def test_rejects_inconsistent_threshold_pairs_atomically(self):
        for updates in (
            {"occlusion_dark_value": 220},
            {"occlusion_area_ratio": 0.01, "occlusion_min_region_ratio": 0.02},
            {"roll_review_deg": 30},
            {"mean_dark": 250},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_thresholds(self.defaults, updates)
        self.assertEqual(self.defaults["occlusion_dark_value"], 32)

    def test_restores_legal_pairs_without_order_dependence(self):
        saved = {"occlusion_dark_value": 230, "occlusion_gray_max_value": 240}
        restored = restore_thresholds(self.defaults, saved)
        self.assertEqual(restored["occlusion_dark_value"], 230)
        self.assertEqual(restored["occlusion_gray_max_value"], 240)

    def test_legacy_invalid_settings_fall_back_safely(self):
        restored = restore_thresholds(self.defaults, {
            "occlusion_area_ratio": float("nan"),
            "occlusion_dark_value": 230,
            "occlusion_gray_max_value": 100,
            "baseline_min_good": 2,
            "confidence_review": 0.9,
            "unknown": 42,
        })
        self.assertEqual(restored["occlusion_area_ratio"], 0.35)
        self.assertEqual(restored["occlusion_dark_value"], 32)
        self.assertEqual(restored["occlusion_gray_max_value"], 220)
        self.assertEqual(restored["baseline_min_good"], 8)
        self.assertEqual(restored["confidence_review"], 0.9)
        self.assertNotIn("unknown", restored)
        self.assertEqual(restore_thresholds(self.defaults, []), self.defaults)

    def test_module_switch_defaults_to_qwen_only(self):
        config = Config()
        self.assertEqual(config.module_switches, {
            "occlusion": False,
            "blur": False,
            "qwen_semantic": True,
        })
        self.assertEqual(restore_module_switches({}), config.module_switches)
        self.assertEqual(
            validate_module_switches(
                config.module_switches,
                {"occlusion": True, "blur": True, "qwen_semantic": False},
            ),
            {"occlusion": True, "blur": True, "qwen_semantic": False},
        )
        with self.assertRaises(ValueError):
            validate_module_switches(config.module_switches, {"occlusion": 1})
        with self.assertRaises(ValueError):
            validate_module_switches(config.module_switches, {"unknown": True})


class OcclusionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # main 的模块级初始化也指向临时库，绝不打开工作区真实数据。
        cls.import_tmp = tempfile.TemporaryDirectory()
        root = Path(cls.import_tmp.name)
        initial_config = Config(data_dir=root, db_path=root / "initial.db", demo_mode=True,
                                hik_client_id="", hik_client_secret="", vlm_enabled=False,
                                artifact_root=root / "artifacts")
        with patch("cam_angle.config.get_config", return_value=initial_config):
            cls.main = importlib.import_module("cam_angle.main")
        cls.initial_db = cls.main.db
        cls.original_config = cls.main.config
        cls.original_pipeline = cls.main.pipeline

    @classmethod
    def tearDownClass(cls):
        cls.main.db = cls.initial_db
        cls.main.config = cls.original_config
        cls.main.pipeline = cls.original_pipeline
        cls.initial_db._conn.close()
        cls.import_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = Config(data_dir=self.root, db_path=self.root / "test.db", demo_mode=True,
                             hik_client_id="", hik_client_secret="", region_filter="",
                             vlm_enabled=False, artifact_root=self.root / "artifacts")
        self.config.ensure_dirs()
        self.db = Database(self.config.db_path)
        self.main.config = self.config
        self.main.db = self.db
        self.client = AsgiTestClient(self.main.app)
        self.camera_id = self.db.upsert_camera("TEST", 1, channel_name="测试相机")
        self.image_path = self.root / "sample.jpg"
        self.image_path.write_bytes(b"test capture")

    def tearDown(self):
        self.db._conn.close()
        self.tmp.cleanup()

    def make_capture(self, modules=None, baseline_id=None, decision="pending", batch_id=None):
        return self.db.add_capture(self.camera_id, str(self.image_path), "", "test", 0,
                                   "2026-11-20T12:00:00+08:00", decision=decision,
                                   module_results=modules, baseline_id=baseline_id,
                                   batch_id=batch_id)

    def test_batches_isolate_stats_and_paginate_capture_results(self):
        first_batch = self.db.create_batch("2026-11-20T12:00:00+08:00")
        second_batch = self.db.create_batch("2026-11-20T13:00:00+08:00")
        self.make_capture(decision="pass", batch_id=first_batch)
        self.make_capture(decision="review", batch_id=first_batch)
        self.make_capture(decision="no_signal", batch_id=first_batch)
        self.make_capture(decision="fail", batch_id=second_batch)

        first_stats = self.client.get(f"/api/stats?batch_id={first_batch}").json()
        second_stats = self.client.get(f"/api/stats?batch_id={second_batch}").json()
        self.assertEqual(first_stats["total_captures"], 3)
        self.assertEqual(first_stats["by_decision"]["review"], 1)
        self.assertEqual(second_stats["total_captures"], 1)
        self.assertEqual(second_stats["by_decision"]["fail"], 1)

        page_one = self.client.get(
            f"/api/captures?batch_id={first_batch}&limit=2&offset=0"
        ).json()
        page_two = self.client.get(
            f"/api/captures?batch_id={first_batch}&limit=2&offset=2"
        ).json()
        self.assertEqual(page_one["total"], 3)
        self.assertEqual(len(page_one["rows"]), 2)
        self.assertEqual(len(page_two["rows"]), 1)
        self.assertEqual(page_two["rows"][0]["batch_id"], first_batch)

        cameras = self.client.get(
            f"/api/cameras?batch_id={first_batch}&limit=1&offset=0"
        ).json()
        self.assertEqual(cameras["total"], 1)
        self.assertEqual(cameras["rows"][0]["last_capture"]["batch_id"], first_batch)

        batches = self.client.get("/api/batches").json()["rows"]
        self.assertEqual([row["id"] for row in batches[:2]], [second_batch, first_batch])
        self.assertEqual(batches[1]["capture_count"], 3)

    def test_settings_persist_and_restore_on_startup(self):
        response = self.client.put("/api/settings", json={
            "thresholds": {"occlusion_area_ratio": 0.3},
            "capture_interval": 8,
            "pipeline_mode": "decoupled",
            "capture_concurrency": 16,
            "inference_concurrency": 24,
            "sample_size": 100,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.db.get_setting("thresholds")["occlusion_area_ratio"], 0.3)
        self.assertEqual(self.db.get_setting("pipeline_mode"), "decoupled")
        self.assertEqual(self.db.get_setting("cycle_capture_concurrency"), 16)
        self.assertEqual(self.db.get_setting("cycle_inference_concurrency"), 24)
        self.assertEqual(self.db.get_setting("cycle_sample_size"), 100)
        self.assertEqual(self.db.get_setting("cycle_capture_interval"), 8.0)
        self.config.thresholds = Config().thresholds
        self.config.module_switches = Config().module_switches
        self.config.pipeline_mode = "coupled"
        self.config.capture_concurrency = 8
        self.config.inference_concurrency = 24
        self.config.sample_size = 0
        self.config.capture_interval = 5.0
        asyncio.run(self.main.startup())
        self.assertEqual(self.client.get("/api/settings").json()["thresholds"]["occlusion_area_ratio"], 0.3)
        self.assertEqual(self.client.get("/api/settings").json()["capture_interval"], 8.0)
        restored = self.client.get("/api/settings").json()
        self.assertEqual(restored["pipeline_mode"], "decoupled")
        self.assertEqual(restored["capture_concurrency"], 16)
        self.assertEqual(restored["inference_concurrency"], 24)
        self.assertEqual(restored["sample_size"], 100)

    def test_sample_size_accepts_zero_for_all_and_rejects_invalid_values(self):
        response = self.client.put("/api/settings", json={"sample_size": 0})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["sample_size"], 0)
        self.assertEqual(self.db.get_setting("cycle_sample_size"), 0)
        for value in (-1, True, "100", 100000.5, 100001):
            response = self.client.put("/api/settings", json={"sample_size": value})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("抽样", response.json()["detail"])

    def test_invalid_settings_leave_memory_and_database_unchanged(self):
        original = dict(self.config.thresholds)
        for value in (-1, 25, True, "0.2"):
            response = self.client.put("/api/settings", json={"thresholds": {"occlusion_area_ratio": value}})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("阈值", response.json()["detail"])
            self.assertEqual(self.config.thresholds, original)
            self.assertIsNone(self.db.get_setting("thresholds"))
        response = self.client.put("/api/settings", content=json.dumps({"thresholds": {"occlusion_area_ratio": float("nan")}}),
                                   headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400, response.text)
        for value in (4, 301, True, "8", 8.5):
            response = self.client.put("/api/settings", json={"capture_interval": value})
            if value == 8.5:
                self.assertEqual(response.status_code, 200, response.text)
            else:
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("抓图间隔", response.json()["detail"])

    def test_overlay_served_only_with_metadata_and_existing_file(self):
        capture_id = self.make_capture({"occlusion": {"overlay_available": True}})
        url = f"/api/captures/{capture_id}/occlusion"
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get(f"/api/captures/{capture_id}").json()["occlusion_url"], "")
        overlay_bytes = b"\x89PNG\r\n\x1a\nfixture"
        self.image_path.with_suffix(".occlusion.png").write_bytes(overlay_bytes)
        detail = self.client.get(f"/api/captures/{capture_id}").json()
        self.assertEqual(detail["occlusion_url"], url)
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, overlay_bytes)
        self.assertEqual(self.client.get("/api/captures").json()["rows"][0]["occlusion_url"], url)
        old_capture_id = self.make_capture()
        self.assertEqual(self.client.get(f"/api/captures/{old_capture_id}/occlusion").status_code, 404)
        self.assertEqual(self.client.get("/api/captures/99999/occlusion").status_code, 404)

    def test_baseline_url_remains_bound_to_analyzed_capture(self):
        old_id = self.db.add_baseline(self.camera_id, str(self.root / "old.jpg"))
        capture_id = self.make_capture(baseline_id=old_id)
        self.db.add_baseline(self.camera_id, str(self.root / "new.jpg"))
        expected = f"/api/baselines/{old_id}/image"
        self.assertEqual(self.client.get(f"/api/captures/{capture_id}").json()["baseline_url"], expected)
        self.assertEqual(self.client.get("/api/captures").json()["rows"][0]["baseline_url"], expected)
        no_baseline_id = self.make_capture()
        self.assertEqual(self.client.get(f"/api/captures/{no_baseline_id}").json()["baseline_url"], "")

    def test_failed_review_cannot_replace_baseline(self):
        capture_id = self.make_capture()
        response = self.client.post(f"/api/captures/{capture_id}/review",
                                    json={"decision": "fail", "set_baseline": True})
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.db.get_latest_baseline(self.camera_id))
        self.assertEqual(self.db.get_capture(capture_id)["decision"], "pending")
        with self.db._lock:
            self.assertEqual(self.db._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 0)

    def test_approved_review_can_set_existing_capture_as_baseline(self):
        capture_id = self.make_capture()
        response = self.client.post(f"/api/captures/{capture_id}/review",
                                    json={"decision": "pass", "set_baseline": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.db.get_latest_baseline(self.camera_id)["path"], str(self.image_path))
        self.assertEqual(self.db.get_capture(capture_id)["decision"], "pass")

    def test_missing_capture_cannot_be_set_as_baseline(self):
        capture_id = self.make_capture()
        self.image_path.unlink()
        response = self.client.post(f"/api/captures/{capture_id}/review",
                                    json={"decision": "pass", "set_baseline": True})
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.db.get_latest_baseline(self.camera_id))


if __name__ == "__main__":
    unittest.main()
