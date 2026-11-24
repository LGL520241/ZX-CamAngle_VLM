from __future__ import annotations

from typing import Any, Dict, List

from .config import Config
from .vlm import score_elements


STATUS_MAP = {
    "pass": "pass",
    "fail": "fail",
    "warn": "warn",
    "unknown": "unknown",
}


def _module(results: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = results.get(name)
    return value if isinstance(value, dict) else {}


_ANCHOR_EVIDENCE_TERMS: Dict[str, tuple] = {
    "oil_fryer": ("油锅", "炸锅", "油炸锅"),
    "cash_register": ("收银机", "收银屏", "收银界面"),
    "range_hood": ("油烟机", "排烟罩"),
    "counter": ("操作台", "工作台", "台面", "柜台", "交易台", "收银台", "打包台"),
    "refrigerator": ("冰箱", "冷柜", "冷藏柜", "冷冻柜"),
    "display_case": ("展示柜", "保温柜", "保酥柜"),
    "sink": ("水池", "水槽", "清洗槽"),
    "prep_area": ("备餐区", "切配区", "腌制区"),
    "breading_area": ("裹粉台", "裹粉区"),
    "employee": ("员工", "店员", "工作人员"),
}

_AREA_EVIDENCE_TERMS = (
    "入口",
    "门口",
    "进出通道",
    "人行道",
    "街道",
    "停车区",
    "店外活动区",
    "桌椅",
    "用餐区",
    "货架",
    "仓库",
    "纸箱",
    "货物",
    "储物",
)

_NO_BUSINESS_ANCHOR_PHRASES = (
    "无业务锚点",
    "无完整业务锚点",
    "无明确业务锚点",
    "无可用业务区域",
    "无业务区域",
    "没有业务锚点",
    "看不到业务界面",
    "仅见柜体",
    "柜体内部",
    "设备背板",
    "设备内部",
)

_WORK_INTERFACE_ANCHOR_IDS = {
    "oil_fryer",
    "cash_register",
    "counter",
    "refrigerator",
    "display_case",
    "sink",
    "prep_area",
    "breading_area",
}

_CEILING_FIXTURE_TERMS = (
    "天花板",
    "吊顶",
    "灯具",
    "灯管",
    "排烟罩",
    "通风管",
    "空调出风口",
)

_DINING_EVIDENCE_TERMS = (
    "桌椅",
    "餐桌",
    "用餐区",
    "顾客活动区",
    "顾客通道",
    "用餐通道",
)

_OUTDOOR_EVIDENCE_TERMS = (
    "人行道",
    "街道",
    "马路",
    "停车区",
    "停车线",
    "黄线",
    "店外",
    "门外",
    "入口",
    "通道",
    "电动车",
    "摩托车",
    "路沿",
    "外立面",
)

_CLEAR_ANCHOR_IDS = {
    "oil_fryer",
    "cash_register",
    "counter",
    "refrigerator",
    "display_case",
    "sink",
    "prep_area",
    "breading_area",
}


def _evidence_anchor_ids(semantics: Dict[str, Any]) -> List[str]:
    """Extract explicitly named business anchors from Qwen's evidence text."""
    evidence = " ".join(
        str(semantics.get(key) or "")
        for key in ("view_evidence", "business_scene_summary")
    )
    return sorted(
        anchor_id
        for anchor_id, terms in _ANCHOR_EVIDENCE_TERMS.items()
        if any(term in evidence for term in terms)
    )


def fuse_semantic_decision(
    cv_result: Dict[str, Any],
    semantic_result: Dict[str, Any],
    config: Config,
) -> Dict[str, Any]:
    """融合确定性画质证据与 Qwen 场景语义。

    CV 负责“能否看清”，Qwen 负责“看到了什么、是否正对业务区域”。低分或
    模型不可用时转人工，不把模型自报 confidence 当作统计概率。
    """
    modules = dict(cv_result.get("module_results") or {})
    quality = _module(modules, "quality")
    occlusion = _module(modules, "occlusion")
    blur = _module(modules, "blur")
    rotation = _module(modules, "rotation")
    semantics = semantic_result.get("data") or {}
    review_score = float(config.thresholds.get("vlm_review_score", config.vlm_review_score))
    min_element_confidence = float(
        config.thresholds.get("vlm_min_element_confidence", config.vlm_min_element_confidence)
    )
    element = score_elements(semantics, min_element_confidence)

    reasons: List[str] = []
    hard_failures: List[str] = []
    reviews: List[str] = []
    quality_low_texture_warn = False

    if quality.get("status") == "fail":
        hard_failures.append(f"quality: {quality.get('reason', '画面不可用')}")
    if blur.get("status") == "fail":
        hard_failures.append(f"blur: {blur.get('reason', '画面模糊')}")
    if rotation.get("status") == "fail":
        hard_failures.append(f"rotation: {rotation.get('reason', '画面严重倾斜')}")
    # 轻微倾斜只记录指标，不影响最终判定。
    for name in ("quality", "blur"):
        module = _module(modules, name)
        if module.get("status") in {"warn", "unknown"}:
            quality_warn_reason = str(module.get("reason") or "")
            if (
                name == "quality"
                and module.get("status") == "warn"
                and "低纹理" in quality_warn_reason
            ):
                # A low-texture scalar can be a false wall warning for flat,
                # reflective counters. Delay it until semantic coverage is known.
                quality_low_texture_warn = True
            else:
                reviews.append(f"{name}: {quality_warn_reason or '证据不足'}")

    occlusion_ratio = float((occlusion.get("metrics") or {}).get("occlusion_ratio", 0.0))
    cv_occlusion_candidate = occlusion.get("status") == "fail"
    if cv_occlusion_candidate:
        hard_failures.append(
            f"occlusion: 遮挡面积候选占 {occlusion_ratio:.1%}，"
            f"超过 {config.thresholds.get('occlusion_area_ratio', 0.35):.1%} 阈值"
        )
    elif occlusion.get("status") == "warn":
        reviews.append(
            f"occlusion: 检测到小部分遮挡候选（{occlusion_ratio:.1%}），需人工确认是否影响人员行为判断"
        )
    vlm_occlusion = semantics.get("occlusion", "unknown")
    semantic_occlusion_ratio = float(semantics.get("occlusion_area_ratio", 0.0))
    semantic_occlusion_affects_subject = bool(
        semantics.get("occlusion_affects_subject")
    )
    vlm_outdoor = bool(semantics.get("is_store_outdoor"))
    vlm_entrance_flow = bool(semantics.get("is_entrance_flow")) or (
        semantics.get("camera_type") == "store_entrance"
    )
    vlm_conf = float(semantics.get("confidence", 0.0))
    camera_type = semantics.get("camera_type", "unknown")
    semantic_available = semantic_result.get("status") in {"pass", "warn"} and bool(semantics)
    semantic_blur = semantics.get("blur", "unknown")
    semantic_blur_scope = semantics.get("blur_scope", "unknown")
    semantic_occlusion_scope = semantics.get("occlusion_scope", "unknown")
    operation_visibility = semantics.get("operation_visibility", "unknown")
    hand_action_visible = bool(semantics.get("hand_action_visible"))
    monitoring_subject_affected = bool(
        semantics.get("monitoring_subject_affected")
    )
    verified_obstruction = bool(semantics.get("obstruction_verified"))
    view_valid = bool(semantics.get("view_valid"))
    view_issue = semantics.get("view_issue", "unknown")
    primary_subject_central = bool(semantics.get("primary_subject_central"))
    subject_coverage = semantics.get("subject_coverage", "none")
    target_area_verified = bool(semantics.get("target_area_visible")) or (
        element["score"] >= review_score
        or view_valid
    )
    semantic_evidence_text = " ".join(
        str(semantics.get(key) or "")
        for key in (
            "view_evidence",
            "occlusion_evidence",
            "business_scene_summary",
        )
    )
    evidence_anchor_ids = _evidence_anchor_ids(semantics)
    ceiling_fixture_only_evidence = (
        camera_type in {"kitchen", "cashier", "kitchen_cashier", "other_indoor"}
        and any(term in semantic_evidence_text for term in _CEILING_FIXTURE_TERMS)
        and not _WORK_INTERFACE_ANCHOR_IDS.intersection(evidence_anchor_ids)
    )
    no_anchor_check_text = (
        semantic_evidence_text
        .replace("无业务锚点冲突", "")
        .replace("无业务锚点问题", "")
    )
    no_business_anchor_evidence = any(
        phrase in no_anchor_check_text
        for phrase in _NO_BUSINESS_ANCHOR_PHRASES
    )
    if no_business_anchor_evidence or ceiling_fixture_only_evidence:
        target_area_verified = False
        view_valid = False
        view_issue = "wall_ceiling" if ceiling_fixture_only_evidence else "unrecognized"
        subject_coverage = "none"
        monitoring_subject_affected = True
        semantics["camera_type"] = "unknown"
        semantics["target_area_visible"] = False
        semantics["view_valid"] = False
        semantics["view_issue"] = view_issue
        semantics["subject_coverage"] = "none"
        semantics["monitoring_subject_affected"] = True
        camera_type = "unknown"

    outdoor_evidence_hits = {
        term for term in _OUTDOOR_EVIDENCE_TERMS
        if term in semantic_evidence_text
    }
    outdoor_anchor_clear = (
        semantic_available
        and camera_type == "unknown"
        and semantics.get("outdoor_view") == "proper"
        and len(outdoor_evidence_hits) >= 2
        and not no_business_anchor_evidence
        and vlm_occlusion not in {"severe", "unknown"}
        and not semantic_occlusion_affects_subject
        and semantic_blur not in {"severe", "unknown"}
    )
    if outdoor_anchor_clear:
        camera_type = "store_outdoor"
        vlm_outdoor = True
        target_area_verified = True
        view_valid = True
        view_issue = "proper"
        subject_coverage = "medium"
        monitoring_subject_affected = False
        semantics["camera_type"] = camera_type
        semantics["is_store_outdoor"] = True
        semantics["target_area_visible"] = True
        semantics["view_valid"] = True
        semantics["view_issue"] = "proper"
        semantics["subject_coverage"] = subject_coverage
        semantics["monitoring_subject_affected"] = False

    clear_anchor_evidence_text = str(semantics.get("view_evidence") or "")
    clear_anchor_language = (
        any(term in clear_anchor_evidence_text for term in ("清晰", "完整", "可辨"))
        and not any(
            term in clear_anchor_evidence_text
            for term in ("模糊", "虚焦", "不清", "不足", "局部", "边缘", "遮挡")
        )
    )
    clear_anchor_count = len(
        set(element["hit_ids"]).intersection(_CLEAR_ANCHOR_IDS)
    )
    contradictory_clear_anchor = (
        semantic_available
        and camera_type in {"kitchen", "cashier", "kitchen_cashier"}
        and bool(semantics.get("target_area_visible"))
        and subject_coverage in {"medium", "large"}
        and vlm_occlusion not in {"severe", "unknown"}
        and not semantic_occlusion_affects_subject
        and clear_anchor_language
        and clear_anchor_count >= 2
    )
    if contradictory_clear_anchor:
        # Evidence naming multiple complete anchors is more specific than a
        # generic severe-blur label. Reconcile the contradictory enum fields.
        semantic_blur = "slight"
        semantic_blur_scope = "localized"
        monitoring_subject_affected = False
        operation_visibility = "not_applicable"
        view_valid = True
        view_issue = "proper"
        semantics["blur"] = semantic_blur
        semantics["blur_scope"] = semantic_blur_scope
        semantics["monitoring_subject_affected"] = False
        semantics["operation_visibility"] = operation_visibility
        semantics["view_valid"] = True
        semantics["view_issue"] = "proper"

    dining_layout_clear = (
        semantic_available
        and camera_type == "dining"
        and bool(semantics.get("target_area_visible"))
        and any(
            term in semantic_evidence_text
            for term in _DINING_EVIDENCE_TERMS
        )
        and not no_business_anchor_evidence
        and vlm_occlusion not in {"severe", "unknown"}
        and not semantic_occlusion_affects_subject
        and semantic_occlusion_scope not in {"critical", "unknown"}
        and not monitoring_subject_affected
        and semantic_blur not in {"severe", "unknown"}
        and not (
            semantic_blur_scope == "global"
            and semantic_blur != "none"
        )
        and (vlm_conf <= 0.0 or vlm_conf >= 0.55)
    )
    if dining_layout_clear:
        view_valid = True
        view_issue = "proper"
        if subject_coverage in {"none", "small"}:
            subject_coverage = "medium"
        vlm_occlusion = "none"
        semantic_occlusion_ratio = 0.0
        semantic_occlusion_scope = "none"
        semantic_occlusion_affects_subject = False
        monitoring_subject_affected = False
        semantics["view_valid"] = True
        semantics["view_issue"] = "proper"
        semantics["subject_coverage"] = subject_coverage
        semantics["occlusion"] = "none"
        semantics["occlusion_area_ratio"] = 0.0
        semantics["occlusion_scope"] = "none"
        semantics["occlusion_affects_subject"] = False
        semantics["monitoring_subject_affected"] = False

    blocking_occlusion = verified_obstruction or (
        semantic_occlusion_scope in {"critical", "unknown"}
        and (
            semantic_occlusion_affects_subject
            or (
                vlm_occlusion == "severe"
                and vlm_conf >= 0.60
                and (monitoring_subject_affected or operation_visibility == "obscured")
            )
        )
    )
    blocking_blur = (
        semantic_blur_scope in {"global", "unknown"}
        and (semantic_blur == "severe" or operation_visibility == "obscured")
    )
    unusable_severe_blur = (
        semantic_blur == "severe"
        and semantic_blur_scope == "global"
        and monitoring_subject_affected
        and not hand_action_visible
    )
    quality_guard = _module(modules, "quality_guard")
    if quality_guard.get("available"):
        guard_occlusion = quality_guard.get("occlusion") or {}
        guard_blur = quality_guard.get("blur") or {}
        central_guard = quality_guard.get("central") or {}
        central_occlusion = central_guard.get("occlusion") or {}
        central_blur = central_guard.get("blur") or {}
        if (
            blocking_occlusion
            and not verified_obstruction
            and central_occlusion.get("status") in {"pass", "warn"}
        ):
            blocking_occlusion = False
        central_blur_status = central_blur.get("status")
        if (
            blocking_blur
            and central_blur_status == "pass"
            and not unusable_severe_blur
        ):
            blocking_blur = False
        elif (
            blocking_blur
            and central_blur_status in {"warn", "unknown"}
            and not unusable_severe_blur
        ):
            # Only a passing central check can dismiss a semantic severe blur.
            # Warn/unknown means the CV evidence is inconclusive, not harmless.
            blocking_blur = False
        business_subject_visible = (
            element["score"] >= review_score
            and (
                central_occlusion.get("status") in {"pass", "warn"}
                or guard_blur.get("status") in {"pass", "unknown"}
            )
        )
    else:
        business_subject_visible = False

    storage_evidence_text = " ".join(
        str(semantics.get(key) or "")
        for key in (
            "view_evidence",
            "occlusion_evidence",
            "business_scene_summary",
        )
    )
    storage_layout_clear = (
        semantic_available
        and camera_type == "storage"
        and bool(semantics.get("target_area_visible"))
        and not no_business_anchor_evidence
        and any(
            term in storage_evidence_text
            for term in ("仓库", "货架", "纸箱", "货物", "储物")
        )
        and vlm_occlusion not in {"severe", "unknown"}
        and not semantic_occlusion_affects_subject
        and semantic_occlusion_scope not in {"critical", "unknown"}
        and not monitoring_subject_affected
        and semantic_blur not in {"severe", "unknown"}
        and not (
            semantic_blur_scope == "global"
            and semantic_blur != "none"
        )
        and (vlm_conf <= 0.0 or vlm_conf >= 0.55)
    )
    if storage_layout_clear:
        # Boxes and stock are the monitored content of a storage room, not a
        # middle layer. Normalize contradictory "small/unknown" fields only
        # when the model also confirms no actual subject impact.
        view_valid = True
        view_issue = "proper"
        if subject_coverage in {"none", "small"}:
            subject_coverage = "medium"
        vlm_occlusion = "none"
        semantic_occlusion_ratio = 0.0
        semantic_occlusion_scope = "none"
        semantic_occlusion_affects_subject = False
        monitoring_subject_affected = False
        semantics["target_area_visible"] = True
        semantics["view_valid"] = True
        semantics["view_issue"] = "proper"
        semantics["subject_coverage"] = subject_coverage
        semantics["occlusion"] = "none"
        semantics["occlusion_area_ratio"] = 0.0
        semantics["occlusion_scope"] = "none"
        semantics["occlusion_affects_subject"] = False
        semantics["monitoring_subject_affected"] = False

    semantic_anchor_ids = sorted(
        set(evidence_anchor_ids) | set(element["hit_ids"])
    )
    expected_non_employee = {
        anchor_id
        for anchor_id in element["expected_ids"]
        if anchor_id != "employee"
    }
    has_specific_workspace_anchor = bool(
        expected_non_employee.intersection(semantic_anchor_ids)
    )
    evidence_text = " ".join(
        str(semantics.get(key) or "")
        for key in ("view_evidence", "business_scene_summary")
    )
    has_area_anchor = any(term in evidence_text for term in _AREA_EVIDENCE_TERMS)
    has_required_anchor = (
        has_specific_workspace_anchor
        if camera_type in {"kitchen", "cashier", "kitchen_cashier"}
        else has_area_anchor
    )
    semantic_coverage_clear = (
        semantic_available
        and has_required_anchor
        and bool(semantics.get("target_area_visible"))
        and view_valid
        and view_issue == "proper"
        and subject_coverage in {"medium", "large"}
        and not monitoring_subject_affected
        and vlm_occlusion not in {"severe", "unknown"}
        and not semantic_occlusion_affects_subject
        and semantic_blur not in {"severe", "unknown"}
        and not (
            semantic_blur_scope == "global"
            and semantic_blur != "none"
        )
        and operation_visibility != "obscured"
        and (vlm_conf <= 0.0 or vlm_conf >= 0.55)
    )
    if quality_low_texture_warn and not semantic_coverage_clear:
        reviews.append(
            f"quality: {quality.get('reason', '画面低纹理，证据不足')}"
        )

    invalid_subject_view = (
        semantic_available
        and not target_area_verified
        and element["score"] < review_score
        and (
            monitoring_subject_affected
            or vlm_occlusion in {"partial", "severe"}
            or (semantic_blur == "severe" and semantic_blur_scope == "global")
        )
    )
    if invalid_subject_view:
        hard_failures.append(
            "semantic_subject: 目标监控区域不可见、业务元素不足，"
            "且画面存在严重模糊、遮挡或遮挡未知"
        )
    unusable_partial_view = (
        semantic_available
        and not view_valid
        and subject_coverage in {"none", "small"}
        and vlm_occlusion in {"partial", "severe"}
        and semantic_occlusion_ratio >= 0.15
        and view_issue in {"obstructed", "too_close", "wrong_area"}
        and element["score"] < 0.30
    )
    if unusable_partial_view:
        hard_failures.append(
            "camera_view: 业务区域仅剩局部，且中层遮挡已影响可用覆盖"
            + (f"（{semantics.get('view_evidence')}）" if semantics.get("view_evidence") else "")
        )

    # 主业务区域可以位于中央、下部或侧中部。不能用单个
    # primary_subject_central 缺省值推翻已有的有效覆盖证据。
    positive_business_coverage = (
        target_area_verified
        or view_valid
        or primary_subject_central
        or subject_coverage in {"medium", "large"}
    )
    uncertain_business_view = camera_type != "unknown" and not positive_business_coverage
    obvious_invalid_view = (
        uncertain_business_view
        and not view_valid
        and subject_coverage == "none"
        and element["score"] < review_score
        and view_issue in {"wall_ceiling", "too_close", "obstructed", "unrecognized"}
    )
    if semantic_available and obvious_invalid_view:
        hard_failures.append(
            f"camera_view: 核心业务区域未有效覆盖（{view_issue}，"
            f"有效覆盖={view_valid}，覆盖={subject_coverage}）"
            + (f"（{semantics.get('view_evidence')}）" if semantics.get("view_evidence") else "")
        )
    elif semantic_available and uncertain_business_view:
        reviews.append(
            f"camera_view: 核心业务区域覆盖不明确（{view_issue}），需人工复核"
        )

    semantic_status = "unknown"
    semantic_confidence = vlm_conf
    if not semantic_available:
        reviews.append(f"semantic: {semantic_result.get('reason', 'Qwen 结果不可用')}")
    elif camera_type == "unknown" and not bool(semantics.get("target_area_visible")):
        hard_failures.append(
            "semantic_scene: Qwen 无法识别有效门店场景，画面可能被近距离物体遮挡"
            + (
                f"（{semantics.get('business_scene_summary')}）"
                if semantics.get("business_scene_summary")
                else ""
            )
        )
        semantic_status = "fail"
    elif verified_obstruction:
        hard_failures.append(
            "occlusion_verified: 遮挡专项复核确认近景物体大面积遮挡核心业务区域"
            + (
                f"（{semantics.get('occlusion_evidence')}）"
                if semantics.get("occlusion_evidence")
                else ""
            )
        )
        semantic_status = "fail"
    elif blocking_occlusion:
        hard_failures.append(
            f"occlusion: 遮挡面积估计 {semantic_occlusion_ratio:.1%}，"
            + "且已影响监控主体或人员行为"
            + (f"（{semantics.get('occlusion_evidence')}）" if semantics.get("occlusion_evidence") else "")
        )
        semantic_status = "fail"
    elif blocking_blur:
        hard_failures.append(
            "blur: Qwen 判断画面模糊或遮挡已经影响人员行为判断"
            + (f"（{semantics.get('blur_evidence')}）" if semantics.get("blur_evidence") else "")
        )
        semantic_status = "fail"
    elif vlm_entrance_flow:
        outdoor_view = semantics.get("outdoor_view", "unknown")
        angle_quality = semantics.get("angle_quality", "unknown")
        if (
            outdoor_view in {"facing_sky", "facing_wall"}
            and not target_area_verified
            and monitoring_subject_affected
        ):
            hard_failures.append(
                f"entrance_angle: 门口人流摄像头{ {'facing_sky': '对天', 'facing_wall': '对墙'}[outdoor_view] }，无法可靠识别进出人流"
            )
            semantic_status = "fail"
        elif (
            target_area_verified
            and not blocking_occlusion
            and not blocking_blur
        ):
            # 门口/客流场景不要求油锅、收银机等餐饮元素。
            semantic_status = "pass"
        else:
            reviews.append("entrance_angle: 门口进出通道覆盖情况不明确，需人工确认")
            semantic_status = "warn"
    elif vlm_outdoor:
        outdoor_view = semantics.get("outdoor_view", "unknown")
        angle_quality = semantics.get("angle_quality", "unknown")
        if (
            outdoor_view in {"facing_sky", "facing_wall"}
            and not target_area_verified
            and monitoring_subject_affected
        ):
            hard_failures.append(
                f"outdoor_angle: 店外摄像头{ {'facing_sky': '对天', 'facing_wall': '对墙'}[outdoor_view] }，角度不合适"
            )
            semantic_status = "fail"
        elif (
            target_area_verified
            and not blocking_occlusion
            and not blocking_blur
        ):
            semantic_status = "pass"
        else:
            reviews.append("outdoor_angle: 店外摄像头朝向证据不足，需人工确认")
            semantic_status = "warn"
    elif vlm_occlusion == "unknown":
        reviews.append("occlusion: 目标区域可见，但遮挡影响仍无法确认")
        semantic_status = "warn"
    elif camera_type in {"dining", "storage", "other_indoor"}:
        # These scenes do not require element scoring, but they still need a
        # consistent visible/valid/proper coverage result.
        if (
            bool(semantics.get("target_area_visible"))
            and view_valid
            and view_issue == "proper"
            and subject_coverage in {"medium", "large"}
        ):
            semantic_status = "pass"
        else:
            reviews.append(
                f"camera_view: 场景覆盖字段不一致（{view_issue}，"
                f"有效覆盖={view_valid}，覆盖={subject_coverage}），需人工复核"
            )
            semantic_status = "warn"
    elif not target_area_verified:
        reviews.append("semantic: Qwen 未确认拍摄到目标业务区域")
        semantic_status = "warn"
    elif element["score"] >= review_score:
        # confidence=0 常见于截断修复后缺失的 confidence 字段，
        # 不能把它当成模型明确给出的低置信度。
        if vlm_conf <= 0.0 or vlm_conf >= 0.55:
            semantic_status = "pass"
        else:
            reviews.append(
                f"element_match: 业务元素命中率 {element['score']:.0%}，"
                f"已达到 {review_score:.0%} 阈值；但模型置信度仅 {vlm_conf:.0%}，"
                "低于 55%，需人工复核"
            )
            semantic_status = "warn"
    elif semantic_coverage_clear:
        # Qwen can identify a complete workspace in evidence while omitting
        # the redundant element list. Treat that as clear coverage rather
        # than sending an otherwise valid frame to manual review.
        semantic_status = "pass"
    else:
        reviews.append(
            f"element_match: 业务元素命中率仅 {element['score']:.0%}，"
            f"低于 {review_score:.0%} 复核阈值"
        )
        semantic_status = "warn"

    modules["semantic"] = {
        "status": semantic_status,
        "confidence": round(semantic_confidence, 3),
        "reason": semantics.get("business_scene_summary") or semantic_result.get("reason", ""),
        "camera_type": camera_type,
        "is_store_outdoor": vlm_outdoor,
        "is_entrance_flow": vlm_entrance_flow,
        "target_area_visible": bool(semantics.get("target_area_visible")),
        "occlusion": vlm_occlusion,
        "occlusion_area_ratio": semantic_occlusion_ratio,
        "occlusion_affects_subject": semantic_occlusion_affects_subject,
        "occlusion_scope": semantic_occlusion_scope,
        "blur": semantic_blur,
        "blur_evidence": semantics.get("blur_evidence", ""),
        "blur_scope": semantic_blur_scope,
        "operation_visibility": operation_visibility,
        "monitoring_subject_affected": monitoring_subject_affected,
        "obstruction_verified": verified_obstruction,
        "view_valid": view_valid,
        "view_issue": view_issue,
        "primary_subject_central": primary_subject_central,
        "subject_coverage": subject_coverage,
        "view_evidence": semantics.get("view_evidence", ""),
        "outdoor_view": semantics.get("outdoor_view"),
        "angle_quality": semantics.get("angle_quality"),
        "angle_evidence": semantics.get("angle_evidence", ""),
        "occlusion_evidence": semantics.get("occlusion_evidence", ""),
        "visible_elements": semantics.get("visible_elements", []),
        "anchor_evidence_ids": semantic_anchor_ids,
        "business_anchor_confirmed": semantic_coverage_clear,
        "storage_layout_confirmed": storage_layout_clear,
        "no_business_anchor_evidence": no_business_anchor_evidence,
        "ceiling_fixture_only_evidence": ceiling_fixture_only_evidence,
        "contradictory_clear_anchor": contradictory_clear_anchor,
        "dining_layout_confirmed": dining_layout_clear,
        "outdoor_anchor_confirmed": outdoor_anchor_clear,
        "metrics": semantic_result.get("metrics", {}),
    }
    element_applicable = camera_type not in {
        "store_entrance",
        "dining",
        "storage",
        "other_indoor",
    }
    modules["element_match"] = {
        "status": (
            "skipped"
            if not element_applicable
            else (
                "pass"
                if element["score"] >= review_score or semantic_coverage_clear
                else "warn"
            )
        ),
        "confidence": round(min(1.0, max(vlm_conf, element["score"])), 3),
        "reason": (
            "当前场景不参与元素命中评分"
            if not element_applicable
            else (
                "文字证据确认完整业务锚点："
                + "、".join(semantic_anchor_ids)
                if semantic_coverage_clear and element["score"] < review_score
                else f"命中 {element['hit_count']} 个业务元素："
                + (
                    "、".join(element["hit_labels"])
                    if element["hit_labels"]
                    else "无"
                )
            )
        ),
        "score": element["score"],
        "applicable": element_applicable,
        "anchor_evidence_ids": semantic_anchor_ids,
        **element,
    }
    # exposure fail can be found in quality, but VLM still documents the scene.
    if semantic_status == "fail" and not hard_failures:
        hard_failures.append("semantic: Qwen 语义判定不合格")

    if hard_failures:
        decision = "fail"
        reasons = hard_failures + reviews
        confidence = max(
            [float(m.get("confidence", 0.0)) for m in (quality, blur, rotation)]
            + [vlm_conf, 0.8]
        )
    elif reviews:
        decision = "review"
        reasons = reviews
        confidence = min(0.75, max(0.5, vlm_conf))
    else:
        decision = "pass"
        reasons = []
        confidence = min(
            [vlm_conf]
            + [float(m.get("confidence", 0.0)) for m in (quality, blur, rotation) if m]
        )

    operation_impact = "unknown"
    if hand_action_visible:
        operation_impact = "none"
    elif operation_visibility == "clear":
        operation_impact = "none"
    elif operation_visibility == "partial":
        operation_impact = "minor"
    elif operation_visibility == "obscured":
        operation_impact = "major"
    elif operation_visibility == "not_applicable":
        operation_impact = "major" if semantic_occlusion_affects_subject else "none"

    # 场景方向错误和无效场景属于硬失败。遮挡、模糊、倾斜或画质问题
    # 只要员工手部和具体操作仍然清楚，就按监控主体未受影响处理。
    scene_failure = any(
        reason.startswith((
            "outdoor_angle:",
            "entrance_angle:",
            "semantic_scene:",
            "semantic_subject:",
            "occlusion_verified:",
        ))
        for reason in hard_failures
    )
    override_applied = False
    if decision == "fail" and semantic_available and not scene_failure:
        technical_failure = bool(hard_failures) and all(
            reason.startswith(("blur:", "quality:", "occlusion:"))
            for reason in hard_failures
        )
        if (
            hand_action_visible
            or (
                business_subject_visible
                and technical_failure
                and not unusable_severe_blur
            )
        ) and target_area_verified:
            decision = "pass"
            reasons = []
            confidence = max(0.55, min(0.95, vlm_conf))
            override_applied = True
            modules["semantic"]["status"] = "pass"
            modules["semantic"]["override_reason"] = (
                "原画质或遮挡检查不合格，但监控主体和关键业务元素清楚可见，最终按合格处理"
            )
    modules["semantic"]["hand_action_visible"] = hand_action_visible
    modules["semantic"]["operation_impact"] = operation_impact
    modules["semantic"]["final_decision"] = decision

    metrics = dict(cv_result.get("metrics") or {})
    metrics.update(semantic_result.get("metrics") or {})
    metrics.update({
        "vlm_camera_type": camera_type,
        "vlm_is_store_outdoor": vlm_outdoor,
        "vlm_is_entrance_flow": vlm_entrance_flow,
        "vlm_target_area_visible": bool(semantics.get("target_area_visible")),
        "vlm_occlusion": vlm_occlusion,
        "vlm_occlusion_area_ratio": semantic_occlusion_ratio,
        "vlm_occlusion_affects_subject": semantic_occlusion_affects_subject,
        "vlm_occlusion_scope": semantic_occlusion_scope,
        "vlm_blur": semantic_blur,
        "vlm_blur_scope": semantic_blur_scope,
        "vlm_target_area_verified": target_area_verified,
        "vlm_operation_visibility": operation_visibility,
        "vlm_operation_impact": operation_impact,
        "vlm_monitoring_subject_affected": monitoring_subject_affected,
        "vlm_obstruction_verified": verified_obstruction,
        "vlm_view_valid": view_valid,
        "vlm_view_issue": view_issue,
        "vlm_primary_subject_central": primary_subject_central,
        "vlm_subject_coverage": subject_coverage,
        "final_override_applied": override_applied,
        "vlm_outdoor_view": semantics.get("outdoor_view"),
        "vlm_angle_quality": semantics.get("angle_quality"),
        "element_score": element["score"],
        "element_hit_count": element["hit_count"],
        "element_hit_ids": element["hit_ids"],
        "anchor_evidence_ids": semantic_anchor_ids,
        "business_anchor_confirmed": semantic_coverage_clear,
        "storage_layout_confirmed": storage_layout_clear,
        "no_business_anchor_evidence": no_business_anchor_evidence,
        "ceiling_fixture_only_evidence": ceiling_fixture_only_evidence,
        "contradictory_clear_anchor": contradictory_clear_anchor,
        "dining_layout_confirmed": dining_layout_clear,
        "outdoor_anchor_confirmed": outdoor_anchor_clear,
    })
    return {
        "decision": decision,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "reasons": reasons,
        "metrics": metrics,
        "module_results": modules,
        "baseline_id": cv_result.get("baseline_id"),
    }
