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
    normalize_obstruction_verification,
    normalize_semantic,
    parse_json_content,
    repair_truncated_json,
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

    def test_repairs_json_truncated_inside_evidence(self):
        result = repair_truncated_json(
            '{"camera_type":"kitchen","view_valid":true,'
            '"visible_elements":[{"id":"counter","visible":true,'
            '"confidence":0.8,"evidence":"可见操作台'
        )
        semantic = normalize_semantic(result)
        counter = next(
            item for item in semantic["visible_elements"] if item["id"] == "counter"
        )
        self.assertTrue(counter["visible"])

    def test_repairs_json_truncated_after_field_colon(self):
        result = repair_truncated_json(
            '{"camera_type":"kitchen","view_valid":true,'
            '"business_scene_summary":"",'
            '"confidence":'
        )
        semantic = normalize_semantic(result)
        self.assertEqual(semantic["camera_type"], "kitchen")
        self.assertEqual(semantic["confidence"], 0.0)

    def test_schema_lists_all_canonical_elements(self):
        properties = RESPONSE_SCHEMA["properties"]["visible_elements"]["items"]["properties"]
        self.assertIn("oil_fryer", properties["id"]["enum"])
        self.assertIn("cash_register", properties["id"]["enum"])
        self.assertIn("refrigerator", properties["id"]["enum"])
        self.assertIn("breading_area", properties["id"]["enum"])
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

    def test_large_foreground_keeps_explicit_scene_usable_verdict(self):
        result = normalize_obstruction_verification({
            "foreground_area_ratio": 0.45,
            "target_overlap_ratio": 0.35,
            "target_area_ratio": 0.25,
            "middle_layer_intrusion": False,
            "occupies_primary_view": True,
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": True,
            "target_position": "right_side",
            "obstruction_type": "none",
            "evidence": "前方箱体占据下部，后方员工可见",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["affects_core_area"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])

    def test_obstruction_small_peripheral_foreground_remains_usable(self):
        result = normalize_obstruction_verification({
            "foreground_area_ratio": 0.10,
            "target_overlap_ratio": 0.05,
            "target_area_ratio": 0.50,
            "middle_layer_intrusion": False,
            "occupies_primary_view": True,
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": True,
            "target_position": "center",
            "obstruction_type": "none",
            "evidence": "边缘轻微干扰",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])

    def test_tiny_partial_target_edge_is_unusable_without_middle_layer(self):
        result = normalize_obstruction_verification({
            "target_area_description": "画面左下角仅见部分操作台面与水槽",
            "foreground_area_ratio": 0.95,
            "target_overlap_ratio": 0.05,
            "target_area_ratio": 0.05,
            "middle_layer_intrusion": False,
            "occupies_primary_view": False,
            "target_position": "left_side",
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": True,
            "obstruction_type": "none",
            "evidence": "其余画面为大面积固定平面，业务区只剩底部局部边缘",
        })
        self.assertTrue(result["lens_obstructed"])
        self.assertTrue(result["affects_core_area"])
        self.assertFalse(result["scene_usable"])

    def test_fixed_ad_screen_is_not_a_middle_layer(self):
        result = normalize_obstruction_verification({
            "target_area_description": "收银台和员工操作区",
            "foreground_area_ratio": 0.45,
            "target_overlap_ratio": 0.25,
            "target_area_ratio": 0.32,
            "middle_layer_intrusion": True,
            "occupies_primary_view": True,
            "target_position": "left_side",
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": False,
            "obstruction_type": "none",
            "evidence": "广告屏为固定结构，非中间层，不构成遮挡；员工操作动作完整可辨。",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])

    def test_near_lens_bracket_not_blocking_visible_core_is_not_obstruction(self):
        result = normalize_obstruction_verification({
            "target_area_description": "右侧收银台和操作面",
            "foreground_area_ratio": 0.45,
            "target_overlap_ratio": 0.25,
            "target_area_ratio": 0.25,
            "middle_layer_intrusion": True,
            "occupies_primary_view": False,
            "target_position": "right_side",
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": False,
            "obstruction_type": "none",
            "evidence": "左侧设备支架不构成遮挡，核心操作面仍清晰可见，员工手部未被遮挡。",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])

    def test_fixed_background_with_visible_employee_hands_is_not_middle_layer(self):
        result = normalize_obstruction_verification({
            "target_area_description": "员工在收银台后操作，操作台面完整可见",
            "foreground_area_ratio": 0.35,
            "target_overlap_ratio": 0.25,
            "target_area_ratio": 0.15,
            "middle_layer_intrusion": True,
            "occupies_primary_view": False,
            "target_position": "left_side",
            "narrow_side_view": False,
            "core_area_split": False,
            "scene_usable": False,
            "obstruction_type": "none",
            "evidence": "右侧墙壁与收银台后方广告牌视觉上更近，但员工手部动作仍可辨识。",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])

    def test_fixed_divider_with_complete_business_unit_is_not_an_obstruction(self):
        result = normalize_obstruction_verification({
            "target_area_description": "右侧为完整业务工作区，含操作台、展示柜和员工",
            "foreground_area_ratio": 0.35,
            "target_overlap_ratio": 0.20,
            "target_area_ratio": 0.25,
            "middle_layer_intrusion": True,
            "occupies_primary_view": False,
            "target_position": "right_side",
            "narrow_side_view": True,
            "core_area_split": True,
            "scene_usable": False,
            "obstruction_type": "pillar",
            "evidence": "中央白色立柱分割画面，但右侧仍为完整业务工作区。",
        })
        self.assertFalse(result["middle_layer_intrusion"])
        self.assertFalse(result["lens_obstructed"])
        self.assertTrue(result["scene_usable"])


class SemanticDecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_kitchen_with_expected_elements_passes(self):
        result = fuse_semantic_decision(cv_result(), semantic(), self.config)
        self.assertEqual(result["decision"], "pass", result)
        self.assertEqual(result["module_results"]["element_match"]["hit_count"], 3)
        self.assertEqual(result["module_results"]["element_match"]["status"], "pass")
        self.assertGreaterEqual(result["metrics"]["element_score"], 0.09)

    def test_breading_area_is_a_kitchen_element(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                visible_elements=[
                    {
                        "id": "breading_area",
                        "visible": True,
                        "confidence": 0.9,
                        "evidence": "裹粉台清晰",
                    }
                ]
            ),
            self.config,
        )
        self.assertIn("breading_area", result["metrics"]["element_hit_ids"])
        self.assertGreaterEqual(result["metrics"]["element_score"], 0.09)

    def test_kitchen_without_visible_elements_goes_to_review(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(visible_elements=[]),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertEqual(result["module_results"]["element_match"]["status"], "warn")

    def test_explicit_workspace_evidence_passes_when_elements_are_omitted(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                visible_elements=[],
                view_evidence="不锈钢操作台与水槽清晰可见",
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertEqual(
            result["module_results"]["element_match"]["status"],
            "pass",
        )
        self.assertTrue(result["metrics"]["business_anchor_confirmed"])
        self.assertIn("counter", result["metrics"]["anchor_evidence_ids"])
        self.assertIn("sink", result["metrics"]["anchor_evidence_ids"])

    def test_low_texture_warning_does_not_review_clear_cashier_view(self):
        cv = cv_result(quality="warn", occlusion="skipped", blur="skipped")
        cv["module_results"]["quality"] = {
            "status": "warn",
            "confidence": 0.62,
            "reason": "画面低纹理，疑似对墙或天花板",
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                camera_type="cashier",
                view_evidence="收银台及操作界面清晰可见",
                subject_coverage="medium",
                operation_visibility="not_applicable",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertEqual(result["reasons"], [])
        self.assertTrue(result["metrics"]["business_anchor_confirmed"])

    def test_low_texture_warning_still_reviews_unclear_view(self):
        cv = cv_result(quality="warn", occlusion="skipped", blur="skipped")
        cv["module_results"]["quality"] = {
            "status": "warn",
            "confidence": 0.62,
            "reason": "画面低纹理，疑似对墙或天花板",
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                target_area_visible=False,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="none",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertTrue(any("低纹理" in reason for reason in result["reasons"]))

    def test_storage_with_visible_contents_is_not_obstructed_by_boxes(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="storage",
                target_area_visible=True,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="small",
                occlusion="partial",
                occlusion_area_ratio=0.35,
                occlusion_scope="none",
                occlusion_affects_subject=False,
                monitoring_subject_affected=False,
                blur="none",
                blur_scope="none",
                view_evidence="堆叠纸箱与设备遮挡核心区域",
                occlusion_evidence="纸箱堆叠部分遮挡操作区",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertTrue(result["metrics"]["storage_layout_confirmed"])
        self.assertEqual(result["metrics"]["vlm_view_valid"], True)
        self.assertEqual(result["metrics"]["vlm_subject_coverage"], "medium")
        self.assertEqual(result["metrics"]["vlm_occlusion"], "none")

    def test_clear_static_anchors_override_contradictory_severe_blur(self):
        cv = cv_result(quality="warn", occlusion="skipped", blur="skipped")
        cv["module_results"]["quality"] = {
            "status": "warn",
            "confidence": 0.62,
            "reason": "画面低纹理，疑似对墙或天花板",
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                camera_type="cashier",
                target_area_visible=True,
                view_valid=False,
                view_issue="blurred",
                subject_coverage="medium",
                occlusion="none",
                blur="severe",
                blur_scope="global",
                monitoring_subject_affected=True,
                operation_visibility="not_applicable",
                view_evidence="收银台及屏幕清晰可见，有操作界面",
                visible_elements=[
                    {"id": "cash_register", "confidence": 0.95},
                    {"id": "display_case", "confidence": 0.90},
                    {"id": "counter", "confidence": 0.90},
                ],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertTrue(result["metrics"]["contradictory_clear_anchor"])
        self.assertEqual(result["metrics"]["vlm_blur"], "slight")

    def test_dining_tables_with_nonimpacting_reflection_pass(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="dining",
                target_area_visible=True,
                view_valid=False,
                view_issue="unknown",
                subject_coverage="small",
                occlusion="partial",
                occlusion_area_ratio=0.35,
                occlusion_scope="none",
                occlusion_affects_subject=False,
                monitoring_subject_affected=False,
                blur="slight",
                blur_scope="localized",
                view_evidence="可见桌椅活动区，含顾客及地面铺装",
                occlusion_evidence="左侧玻璃反射遮挡部分区域",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertTrue(result["metrics"]["dining_layout_confirmed"])
        self.assertEqual(result["metrics"]["vlm_view_valid"], True)
        self.assertEqual(result["metrics"]["vlm_subject_coverage"], "medium")

    def test_unknown_outdoor_view_with_positive_area_evidence_is_normalized(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="unknown",
                is_store_outdoor=False,
                target_area_visible=False,
                view_valid=False,
                view_issue="unrecognized",
                primary_subject_central=False,
                subject_coverage="none",
                occlusion="none",
                occlusion_affects_subject=False,
                occlusion_scope="none",
                monitoring_subject_affected=True,
                blur="none",
                blur_scope="none",
                outdoor_view="proper",
                view_evidence="可见人行道、黄线、摩托车、促销牌和店门",
                business_scene_summary="店外人行道与停车区清晰可辨，无业务锚点冲突",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertTrue(result["metrics"]["outdoor_anchor_confirmed"])
        self.assertEqual(result["metrics"]["vlm_camera_type"], "store_outdoor")
        self.assertTrue(result["metrics"]["vlm_target_area_visible"])

    def test_explicit_no_anchor_evidence_is_not_treated_as_conflict_phrase(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="storage",
                target_area_visible=True,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="small",
                view_evidence="画面为柜体内部，无业务锚点",
                business_scene_summary="柜体内部，无业务锚点",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(result["metrics"]["no_business_anchor_evidence"])

    def test_uncertain_storage_without_layout_evidence_goes_to_review(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="storage",
                target_area_visible=True,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="small",
                occlusion="partial",
                occlusion_area_ratio=0.35,
                occlusion_scope="none",
                occlusion_affects_subject=False,
                monitoring_subject_affected=False,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        self.assertFalse(result["metrics"]["storage_layout_confirmed"])

    def test_explicit_no_anchor_evidence_fails_storage_override(self):
        result = fuse_semantic_decision(
            cv_result(occlusion="skipped", blur="skipped"),
            semantic(
                camera_type="storage",
                target_area_visible=True,
                view_valid=False,
                view_issue="unknown",
                primary_subject_central=False,
                subject_coverage="small",
                occlusion="none",
                occlusion_affects_subject=False,
                occlusion_scope="none",
                monitoring_subject_affected=False,
                blur="none",
                blur_scope="none",
                view_evidence="货架内有纸箱、杂物和包装物，无完整业务锚点",
                business_scene_summary="货架内有杂物，无明确业务锚点或完整业务区",
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(result["metrics"]["no_business_anchor_evidence"])
        self.assertFalse(result["metrics"]["storage_layout_confirmed"])
        self.assertFalse(result["metrics"]["vlm_target_area_visible"])
        self.assertEqual(result["metrics"]["vlm_subject_coverage"], "none")

    def test_ceiling_fixture_without_work_surface_fails_wall_ceiling(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="kitchen",
                target_area_visible=True,
                view_valid=True,
                view_issue="proper",
                subject_coverage="medium",
                view_evidence="完整排烟罩及下方操作区",
                visible_elements=[
                    {
                        "id": "range_hood",
                        "visible": True,
                        "confidence": 0.95,
                        "evidence": "排烟罩完整",
                    },
                    {
                        "id": "counter",
                        "visible": True,
                        "confidence": 0.85,
                        "evidence": "",
                    },
                ],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(result["metrics"]["ceiling_fixture_only_evidence"])
        self.assertEqual(result["metrics"]["vlm_view_issue"], "wall_ceiling")
        self.assertFalse(result["metrics"]["vlm_target_area_visible"])
        self.assertEqual(result["metrics"]["vlm_subject_coverage"], "none")

    def test_ceiling_fixture_with_explicit_work_surface_is_not_rejected(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                camera_type="kitchen",
                view_evidence="完整操作台与排烟罩可见",
                visible_elements=[
                    {
                        "id": "counter",
                        "visible": True,
                        "confidence": 0.9,
                        "evidence": "操作台完整",
                    },
                    {
                        "id": "range_hood",
                        "visible": True,
                        "confidence": 0.9,
                        "evidence": "排烟罩可见",
                    },
                ],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)
        self.assertFalse(result["metrics"]["ceiling_fixture_only_evidence"])

    def test_explicit_anchor_does_not_override_global_blur(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(
                view_evidence="收银台及操作界面可辨",
                camera_type="cashier",
                blur="severe",
                blur_scope="global",
                operation_visibility="not_applicable",
                monitoring_subject_affected=True,
                visible_elements=[],
            ),
            self.config,
        )
        self.assertEqual(result["decision"], "fail", result)
        self.assertFalse(result["metrics"]["business_anchor_confirmed"])

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

    def test_outdoor_severe_semantic_issue_without_anchor_stays_failed(self):
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
        self.assertEqual(result["decision"], "fail", result)
        self.assertTrue(any(reason.startswith("blur:") for reason in result["reasons"]))

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

    def test_unusable_severe_blur_cannot_be_cleared_by_sharp_background(self):
        cv = cv_result(quality="pass", occlusion="skipped", blur="skipped")
        cv["module_results"]["quality_guard"] = {
            "available": True,
            "occlusion": {"status": "pass", "ratio": 0.16},
            "blur": {"status": "pass"},
            "central": {
                "occlusion": {"status": "pass", "ratio": 0.20},
                "blur": {"status": "pass"},
            },
        }
        result = fuse_semantic_decision(
            cv,
            semantic(
                camera_type="dining",
                target_area_visible=True,
                view_valid=False,
                view_issue="blurred",
                primary_subject_central=False,
                subject_coverage="small",
                blur="severe",
                blur_scope="global",
                operation_visibility="not_applicable",
                hand_action_visible=False,
                monitoring_subject_affected=True,
                view_evidence="近景容器遮挡操作区，画面严重虚焦",
                blur_evidence="整体画面严重虚焦，核心区域不可辨",
                visible_elements=[],
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

    def test_high_element_score_with_low_model_confidence_reports_correct_reason(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(confidence=0.4),
            self.config,
        )
        self.assertEqual(result["decision"], "review", result)
        reason = "；".join(result["reasons"])
        self.assertIn("已达到", reason)
        self.assertIn("模型置信度仅 40%", reason)
        self.assertNotIn("低于 9%", reason)

    def test_high_element_score_with_missing_confidence_passes(self):
        result = fuse_semantic_decision(
            cv_result(),
            semantic(confidence=0.0),
            self.config,
        )
        self.assertEqual(result["decision"], "pass", result)

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

    def test_unknown_occlusion_without_target_goes_to_review(self):
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
        self.assertEqual(result["decision"], "review", result)
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
