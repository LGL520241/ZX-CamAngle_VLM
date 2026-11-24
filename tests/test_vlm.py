import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np

from cam_angle.artifacts import ArtifactStore
from cam_angle.config import Config
from cam_angle.decision import fuse_semantic_decision
from cam_angle.vlm import (
    CANONICAL_ELEMENTS,
    RESPONSE_SCHEMA,
    normalize_semantic,
    parse_json_content,
    score_elements,
)


def cv_result(
    *, quality="pass", occlusion="pass", blur="pass", rotation="pass",
    ratio=0.1, black_ratio=0.0,
):
    return {
        "decision": "pass",
        "confidence": 0.9,
        "reasons": [],
        "metrics": {"occlusion_ratio": ratio},
        "module_results": {
            "quality": {"status": quality, "confidence": 0.95, "reason": "画质"},
            "occlusion": {
                "status": occlusion,
                "confidence": 0.9,
                "reason": "遮挡",
                "metrics": {
                    "occlusion_ratio": ratio,
                    "black_region_ratio": black_ratio,
                },
            },
            "blur": {"status": blur, "confidence": 0.9, "reason": "清晰度"},
            "rotation": {"status": rotation, "confidence": 0.9, "reason": "倾斜"},
        },
        "baseline_id": None,
    }


def semantic(**overrides):
    raw = {
        "camera_type": "kitchen",
        "is_store_outdoor": False,
        "is_entrance_flow": False,
        "target_area_visible": True,
        "occlusion": "none",
        "occlusion_area_ratio": 0.0,
        "occlusion_affects_subject": False,
        "occlusion_evidence": "",
        "blur": "none",
        "blur_evidence": "",
        "operation_visibility": "clear",
        "hand_action_visible": False,
        "monitoring_subject_affected": False,
        "view_valid": True,
        "view_issue": "proper",
        "primary_subject_central": True,
        "subject_coverage": "large",
        "view_evidence": "核心业务区可见",
        "outdoor_view": "unknown",
        "angle_quality": "good",
        "angle_evidence": "正对油锅",
        "visible_elements": [
            {"id": "oil_fryer", "visible": True, "confidence": 0.96, "evidence": "炸锅"},
            {"id": "range_hood", "visible": True, "confidence": 0.91, "evidence": "排烟罩"},
            {"id": "counter", "visible": True, "confidence": 0.87, "evidence": "操作台"},
        ],
        "business_scene_summary": "后厨主要操作区清晰可见",
        "confidence": 0.94,
    }
    raw.update(overrides)
    data = normalize_semantic(raw)
    return {
        "status": "pass",
        "confidence": data["confidence"],
        "reason": data["business_scene_summary"],
        "data": data,
        "metrics": {"vlm_latency_ms": 850},
    }


class SemanticNormalizationTests(unittest.TestCase):
    def test_parse_fenced_json_and_normalize_elements(self):
        raw = parse_json_content(
            "```json\n"
            + json.dumps({
                "camera_type": "cashier",
                "is_store_outdoor": False,
                "target_area_visible": True,
                "occlusion": "none",
                "outdoor_view": "unknown",
                "angle_quality": "good",
                "visible_elements": [
                    {"id": "cash_register", "visible": True, "confidence": 1.4},
                    {"id": "invented_object", "visible": True, "confidence": 0.9},
                ],
                "confidence": 0.8,
            })
            + "\n```"
        )
        result = normalize_semantic(raw)
        self.assertEqual(result["camera_type"], "cashier")
        self.assertEqual(len(result["visible_elements"]), len(CANONICAL_ELEMENTS))
        self.assertEqual(score_elements(result)["hit_count"], 1)
        self.assertNotIn("invented_object", {item["id"] for item in result["visible_elements"]})

    def test_schema_lists_all_canonical_elements(self):
        properties = RESPONSE_SCHEMA["properties"]["visible_elements"]["items"]["properties"]
        self.assertIn("oil_fryer", properties["id"]["enum"])
        self.assertIn("cash_register", properties["id"]["enum"])
        self.assertIn("refrigerator", properties["id"]["enum"])
        self.assertIn(
            "store_entrance",
            RESPONSE_SCHEMA["properties"]["camera_type"]["enum"],
        )
        self.assertNotIn(
            "facing_ground",
            RESPONSE_SCHEMA["properties"]["outdoor_view"]["enum"],
        )

    def test_omitted_visible_flag_on_returned_element_counts_as_hit(self):
        result = normalize_semantic({
            "camera_type": "cashier",
            "visible_elements": [
                {"id": "counter", "confidence": 0.9, "evidence": "操作台清楚"},
            ],
            "confidence": 0.8,
        })
        counter = next(item for item in result["visible_elements"] if item["id"] == "counter")
        self.assertTrue(counter["visible"])
        self.assertGreater(score_elements(result)["score"], 0)


class SemanticDecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_kitchen_with_expected_elements_passes(self):
        result = fuse_semantic_decision(cv_result(), semantic(), self.config)
        self.assertEqual(result["decision"], "pass", result)
        self.assertEqual(result["module_results"]["element_match"]["hit_count"], 3)
        self.assertEqual(result["module_results"]["element_match"]["status"], "pass")
        self.assertGreaterEqual(result["metrics"]["element_score"], 0.09)

    def test_kitchen_without_visible_elements_goes_to_review(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(visible_elements=[]),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertEqual(result["module_results"]["element_match"]["status"], "warn")

    def test_slight_rotation_does_not_trigger_review(self):
        result = fuse_semantic_decision(
            cv_result(rotation="warn"),
            semantic(),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertFalse(any("rotation:" in reason for reason in result["reasons"]))

    def test_semantic_severe_blur_fails_directly(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                blur="severe",
                blur_evidence="员工手部操作无法辨认",
                operation_visibility="obscured",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any("影响人员行为判断" in reason for reason in result["reasons"]))

    def test_qwen_large_area_not_affecting_subject_passes(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                occlusion="partial",
                occlusion_area_ratio=0.40,
                occlusion_evidence="左侧被大块门帘遮挡",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertEqual(result["reasons"], [])

    def test_qwen_small_occlusion_not_affecting_subject_passes(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                occlusion="partial",
                occlusion_area_ratio=0.20,
                occlusion_evidence="边缘有少量遮挡",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_qwen_small_occlusion_affecting_subject_fails(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                occlusion="partial",
                occlusion_area_ratio=0.20,
                occlusion_affects_subject=True,
                occlusion_evidence="遮挡员工手部操作",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)

    def test_low_element_score_goes_to_human_review(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                target_area_visible=False,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="none",
                visible_elements=[],
                confidence=0.8,
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertIn("核心业务区域覆盖不明确", "；".join(result["reasons"]))

    def test_outdoor_facing_ground_is_not_an_angle_failure(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="store_outdoor",
                is_store_outdoor=True,
                outdoor_view="facing_ground",
                angle_quality="bad",
                angle_evidence="画面中可见地面",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertNotEqual(result["decision"], "fail", result)
        self.assertFalse(any("对地" in reason for reason in result["reasons"]))

    def test_outdoor_facing_sky_fails(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="store_outdoor",
                is_store_outdoor=True,
                outdoor_view="facing_sky",
                angle_quality="bad",
                target_area_visible=False,
                view_valid=False,
                primary_subject_central=False,
                subject_coverage="none",
                monitoring_subject_affected=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any("对天" in reason for reason in result["reasons"]))

    def test_entrance_flow_does_not_require_business_elements(self):
        result = fuse_semantic_decision(
            cv_result(rotation="warn"),
            semantic(
                camera_type="store_entrance",
                is_store_outdoor=True,
                is_entrance_flow=True,
                outdoor_view="proper",
                angle_quality="good",
                target_area_visible=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertTrue(result["module_results"]["semantic"]["is_entrance_flow"])

    def test_entrance_flow_still_fails_on_severe_rotation(self):
        result = fuse_semantic_decision(
            cv_result(rotation="fail"),
            semantic(
                camera_type="store_entrance",
                is_store_outdoor=False,
                is_entrance_flow=True,
                outdoor_view="proper",
                angle_quality="good",
                target_area_visible=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)

    def test_entrance_flow_facing_sky_fails(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="store_entrance",
                is_store_outdoor=False,
                is_entrance_flow=True,
                outdoor_view="facing_sky",
                angle_quality="bad",
                target_area_visible=False,
                view_valid=False,
                primary_subject_central=False,
                subject_coverage="none",
                monitoring_subject_affected=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(
            any("门口人流摄像头对天" in reason for reason in result["reasons"])
        )

    def test_outdoor_proper_view_passes(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="store_outdoor",
                is_store_outdoor=True,
                outdoor_view="proper",
                angle_quality="good",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_outdoor_severe_semantic_issue_is_cleared_by_passing_guard(self):
        cv = cv_result(occlusion="skipped", blur="skipped")
        cv["module_results"]["quality_guard"] = {
            "available": True,
            "central": {
                "occlusion": {"status": "pass"},
                "blur": {"status": "pass"},
            },
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                camera_type="store_outdoor",
                is_store_outdoor=True,
                target_area_visible=True,
                outdoor_view="proper",
                angle_quality="slightly_off",
                occlusion="severe",
                occlusion_scope="critical",
                occlusion_affects_subject=True,
                blur="severe",
                blur_scope="global",
                operation_visibility="obscured",
                monitoring_subject_affected=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_outdoor_partial_occlusion_not_affecting_subject_passes(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="store_outdoor",
                is_store_outdoor=True,
                occlusion="partial",
                occlusion_evidence="左侧被门体遮挡",
                outdoor_view="proper",
                angle_quality="good",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_occlusion_above_threshold_is_not_overridden_by_operation_visibility(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="fail", ratio=0.50),
            semantic(),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)

    def test_small_partial_occlusion_goes_to_review(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="warn", ratio=0.24),
            semantic(),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertTrue(any("小部分遮挡" in reason for reason in result["reasons"]))

    def test_vlm_unavailable_never_auto_passes(self):
        unavailable = {
            "status": "unknown",
            "confidence": 0.0,
            "reason": "connection refused",
            "data": {},
            "metrics": {},
        }
        result = fuse_semantic_decision(cv_result(), unavailable, self.config)
        self.assertEqual(result["decision"], "review", result)

    def test_quality_failure_passes_when_hand_action_is_clear(self):
        result = fuse_semantic_decision(
            cv_result(quality="fail"),
            semantic(monitoring_subject_affected=True, hand_action_visible=True),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_occlusion_can_be_overridden_when_hand_action_is_clear(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="fail", ratio=0.50),
            semantic(
                occlusion="severe",
                occlusion_area_ratio=0.50,
                occlusion_affects_subject=True,
                hand_action_visible=True,
                target_area_visible=True,
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_severe_semantic_blur_can_be_overridden_by_clear_business_subject(self):
        cv = cv_result(occlusion="skipped", blur="skipped")
        cv["module_results"]["quality_guard"] = {
            "available": True,
            "central": {
                "occlusion": {"status": "pass"},
                "blur": {"status": "fail"},
            },
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                target_area_visible=True,
                blur="severe",
                blur_scope="global",
                operation_visibility="obscured",
                hand_action_visible=False,
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_unusable_severe_blur_with_low_texture_directly_fails(self):
        cv = cv_result(quality="warn", occlusion="skipped", blur="skipped")
        cv["module_results"]["quality_guard"] = {
            "available": True,
            "occlusion": {"status": "fail", "ratio": 0.47},
            "blur": {"status": "unknown"},
            "central": {
                "occlusion": {"status": "fail", "ratio": 0.43},
                "blur": {"status": "unknown"},
            },
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                target_area_visible=True,
                view_valid=True,
                view_issue="blurred",
                blur="severe",
                blur_scope="global",
                operation_visibility="not_applicable",
                hand_action_visible=False,
                monitoring_subject_affected=True,
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any(reason.startswith("blur:") for reason in result["reasons"]))

    def test_verified_near_lens_obstruction_directly_fails(self):
        cv = cv_result(occlusion="skipped", blur="skipped")
        cv["module_results"]["quality_guard"] = {
            "available": True,
            "central": {
                "occlusion": {"status": "pass"},
                "blur": {"status": "pass"},
            },
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                target_area_visible=False,
                view_valid=False,
                view_issue="obstructed",
                primary_subject_central=False,
                subject_coverage="small",
                occlusion="severe",
                occlusion_area_ratio=0.45,
                occlusion_scope="critical",
                occlusion_affects_subject=True,
                monitoring_subject_affected=True,
                obstruction_verified=True,
                hand_action_visible=True,
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(
            any(reason.startswith("occlusion_verified:") for reason in result["reasons"])
        )

    def test_small_partial_view_with_mid_layer_obstruction_fails(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                target_area_visible=True,
                view_valid=False,
                view_issue="obstructed",
                primary_subject_central=False,
                subject_coverage="small",
                occlusion="partial",
                occlusion_area_ratio=0.25,
                occlusion_scope="peripheral",
                occlusion_affects_subject=False,
                monitoring_subject_affected=False,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any("仅剩局部" in reason for reason in result["reasons"]))

    def test_unrecognized_closeup_scene_directly_fails(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="unknown",
                target_area_visible=False,
                monitoring_subject_affected=True,
                business_scene_summary="画面仅显示红色近距离表面",
                visible_elements=[],
                operation_visibility="unknown",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any("无法识别有效门店场景" in reason for reason in result["reasons"]))

    def test_truly_uncovered_kitchen_view_goes_to_review(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                camera_type="kitchen",
                target_area_visible=False,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="none",
                hand_action_visible=False,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertTrue(any("核心业务区域覆盖不明确" in reason for reason in result["reasons"]))

    def test_centered_flag_false_does_not_override_valid_business_coverage(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                camera_type="kitchen",
                target_area_visible=True,
                view_valid=True,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="none",
                hand_action_visible=True,
                operation_visibility="clear",
                visible_elements=[
                    {"id": "counter", "confidence": 0.8},
                    {"id": "refrigerator", "confidence": 0.8},
                ],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_static_cashier_area_is_valid_without_employee(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                camera_type="cashier",
                target_area_visible=True,
                view_valid=True,
                view_issue="proper",
                primary_subject_central=True,
                subject_coverage="medium",
                operation_visibility="not_applicable",
                hand_action_visible=False,
                visible_elements=[
                    {"id": "counter", "confidence": 0.9},
                    {"id": "cash_register", "confidence": 0.8},
                ],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

    def test_obvious_wall_view_without_target_directly_fails(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                camera_type="kitchen",
                target_area_visible=False,
                view_valid=False,
                view_issue="wall_ceiling",
                primary_subject_central=False,
                subject_coverage="none",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)

    def test_unknown_occlusion_without_target_directly_fails(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", ratio=0.0),
            semantic(
                camera_type="kitchen",
                target_area_visible=False,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="none",
                occlusion="unknown",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any("目标区域可见" in reason for reason in result["reasons"]))


class ArtifactStoreTests(unittest.TestCase):
    def test_month_day_and_capture_names(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ArtifactStore(Path(temp))
            cycle = store.begin_cycle("2026-11-20T15:50:19+08:00", 7)
            self.assertEqual(
                cycle.relative_to(Path(temp)).as_posix(),
                "2026/202611/20261120/20261120_155019",
            )
            paths = store.capture_paths(cycle, "2026-11-20T15:50:20.123456+08:00", "ABC/1", 2)
            self.assertTrue(paths["source"].name.startswith("20261120_155020_123__ABC-1_2"))
            self.assertTrue(paths["vlm_request"].name.endswith(".vlm-request.json"))

    def test_prepared_image_is_resized(self):
        from cam_angle.vlm import prepare_image

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.jpg"
            image = np.zeros((1440, 2560, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            path.write_bytes(encoded.tobytes())
            data, meta = prepare_image(path, 1280, 90)
            self.assertGreater(len(data), 0)
            self.assertEqual(meta["input_width"], 1280)
            self.assertEqual(meta["input_height"], 720)


if __name__ == "__main__":
    unittest.main()
