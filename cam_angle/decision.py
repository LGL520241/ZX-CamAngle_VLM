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
            reviews.append(f"{name}: {module.get('reason', '证据不足')}")

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
        if blocking_blur and central_blur_status == "pass":
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

    invalid_subject_view = (
        semantic_available
        and not target_area_verified
        and element["score"] < review_score
        and (
            monitoring_subject_affected
            or vlm_occlusion in {"partial", "severe", "unknown"}
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
        and view_issue in {"obstructed", "too_close", "wrong_area", "unknown"}
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
        # 通道、仓库、用餐区和其他室内区域只看主体与画质，不做元素评分。
        semantic_status = "pass"
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
            else "pass" if element["score"] >= review_score else "warn"
        ),
        "confidence": round(min(1.0, max(vlm_conf, element["score"])), 3),
        "reason": (
            "当前场景不参与元素命中评分"
            if not element_applicable
            else f"命中 {element['hit_count']} 个业务元素："
            + ("、".join(element["hit_labels"]) if element["hit_labels"] else "无")
        ),
        "score": element["score"],
        "applicable": element_applicable,
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
    })
    return {
        "decision": decision,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "reasons": reasons,
        "metrics": metrics,
        "module_results": modules,
        "baseline_id": cv_result.get("baseline_id"),
    }
