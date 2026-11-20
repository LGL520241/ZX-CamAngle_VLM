from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .database import Database


def _read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _quality(gray: np.ndarray, t: Dict[str, float]) -> Dict[str, Any]:
    mean = float(gray.mean())
    std = float(gray.std())
    dark_ratio = float((gray < 10).mean())
    bright_ratio = float((gray > 245).mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())

    metrics = {
        "brightness_mean": round(mean, 2),
        "std": round(std, 2),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "blur_var": round(blur_var, 2),
        "edge_density": round(edge_density, 6),
    }

    if dark_ratio >= t["dark_ratio"] or mean < t["mean_dark"]:
        return {"status": "fail", "confidence": 0.99, "reason": "画面全黑或过暗", "metrics": metrics}
    if bright_ratio >= t["bright_ratio"] or mean > t["mean_bright"]:
        return {"status": "fail", "confidence": 0.99, "reason": "画面全白或过亮", "metrics": metrics}
    if blur_var < t["blur_var"]:
        return {"status": "fail", "confidence": 0.92, "reason": "画面模糊", "metrics": metrics}
    if std < t["std_low"] or edge_density < t["edge_density_low"]:
        return {
            "status": "warn",
            "confidence": 0.62,
            "reason": "画面低纹理，疑似对墙或天花板",
            "metrics": metrics,
        }
    return {"status": "pass", "confidence": 0.95, "reason": "画面可用", "metrics": metrics}


def _rotation(gray: np.ndarray, t: Dict[str, float]) -> Dict[str, Any]:
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=int(w * 0.08),
        maxLineGap=20,
    )
    if lines is None or len(lines) < 3:
        return {"status": "unknown", "confidence": 0.0, "reason": "画面直线特征不足，无法估计倾斜", "angle_deg": 0.0}

    angles: List[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        angles.append(angle)

    horizontal = [a for a in angles if a < 20 or a > 160]
    vertical = [a for a in angles if 70 < a < 110]
    if len(horizontal) >= len(vertical) and horizontal:
        deviations = [a if a < 90 else a - 180 for a in horizontal]
    elif vertical:
        deviations = [a - 90 for a in vertical]
    else:
        deviations = [a if a < 90 else a - 180 for a in angles]

    roll = statistics.median(deviations)
    confidence = min(0.95, 0.45 + len(lines) * 0.02)

    if abs(roll) > t["roll_fail_deg"]:
        return {
            "status": "fail",
            "confidence": round(confidence, 3),
            "reason": f"画面明显倾斜 {roll:.1f}°",
            "angle_deg": round(roll, 2),
        }
    if abs(roll) > t["roll_review_deg"]:
        return {
            "status": "warn",
            "confidence": round(confidence, 3),
            "reason": f"画面轻微倾斜 {roll:.1f}°",
            "angle_deg": round(roll, 2),
        }
    return {
        "status": "pass",
        "confidence": round(confidence, 3),
        "reason": f"画面基本水平（{roll:.1f}°）",
        "angle_deg": round(roll, 2),
    }


def _baseline(
    img: np.ndarray, baseline_path: Optional[Path], t: Dict[str, float]
) -> Tuple[Dict[str, Any], Optional[int]]:
    if baseline_path is None or not baseline_path.exists():
        return {"status": "unknown", "confidence": 0.0, "reason": "无基准图，无法做移位比对", "metrics": {}}, None

    base = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
    if base is None:
        return {"status": "unknown", "confidence": 0.0, "reason": "基准图读取失败", "metrics": {}}, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape != base.shape:
        base = cv2.resize(base, (gray.shape[1], gray.shape[0]))

    corr = 0.0
    if gray.std() > 1e-6 and base.std() > 1e-6:
        corr = float(np.corrcoef(gray.astype(np.float32).ravel(), base.astype(np.float32).ravel())[0, 1])
    if not math.isnan(corr) and corr < 0.7:
        return {
            "status": "fail",
            "confidence": 0.82,
            "reason": f"与基准图整体相似度低（{corr:.2f}），画面疑似被移动或场景变化",
            "metrics": {"correlation": round(corr, 3)},
        }, None

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray, None)
    kp2, des2 = orb.detectAndCompute(base, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return {
            "status": "fail",
            "confidence": 0.70,
            "reason": "画面特征点过少，无法与基准图对齐，疑似被移动",
            "metrics": {},
        }, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < t["baseline_min_good"]:
        return {
            "status": "fail",
            "confidence": 0.78,
            "reason": "与基准图匹配点过少，画面疑似被移动",
            "metrics": {"good_matches": len(good)},
        }, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"status": "unknown", "confidence": 0.0, "reason": "单应矩阵估计失败", "metrics": {}}, None

    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / max(len(good), 1)
    h, w = gray.shape[:2]
    translation = math.hypot(float(H[0, 2]), float(H[1, 2])) / w
    scale = math.sqrt(float(H[0, 0]) ** 2 + float(H[0, 1]) ** 2)
    rot_deg = abs(math.degrees(math.atan2(float(H[1, 0]), float(H[0, 0]))))
    metrics = {
        "correlation": round(corr, 3),
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 3),
        "translation_norm": round(translation, 4),
        "scale": round(scale, 3),
        "rotation_deg": round(rot_deg, 2),
    }

    confidence = round(min(0.97, 0.55 + 0.45 * inlier_ratio), 3)
    if inlier_ratio < t["baseline_inlier_ratio"]:
        return {"status": "fail", "confidence": confidence, "reason": "与基准图对齐率过低，画面可能被移动", "metrics": metrics}, None
    if translation > t["baseline_translation"]:
        return {"status": "fail", "confidence": confidence, "reason": f"画面相对基准偏移较大（{translation:.2f}）", "metrics": metrics}, None
    if rot_deg > t["baseline_rotation_deg"]:
        return {"status": "fail", "confidence": confidence, "reason": f"画面相对基准旋转过大（{rot_deg:.1f}°）", "metrics": metrics}, None
    return {"status": "pass", "confidence": confidence, "reason": "与基准图一致", "metrics": metrics}, None


def _sop(
    metrics: Dict[str, Any],
    template: str,
    rules: Dict[str, float],
) -> Dict[str, Any]:
    violations: List[str] = []
    edge_density = metrics.get("edge_density", 1.0)
    std = metrics.get("std", 255.0)
    if edge_density < rules.get("min_edge_density", 0.004) and std < rules.get("min_std", 18.0):
        violations.append("画面大面积纯色/低纹理，疑似对墙或天花板")

    if violations:
        confidence = 0.80 if len(violations) == 1 else 0.90
        return {"status": "fail", "confidence": confidence, "reason": "；".join(violations), "template": template}
    return {"status": "pass", "confidence": 0.88, "reason": f"SOP 模板「{template}」基础项通过", "template": template}


def _fusion(results: Dict[str, Dict[str, Any]], t: Dict[str, float]) -> Dict[str, Any]:
    fails = [r for r in results.values() if r["status"] == "fail"]
    warns = [r for r in results.values() if r["status"] == "warn"]
    unknowns = [r for r in results.values() if r["status"] == "unknown"]

    if fails:
        confidence = max(r["confidence"] for r in fails)
        decision = "fail" if confidence >= t["confidence_review"] else "review"
    elif unknowns:
        confidence = 0.5
        decision = "review"
    elif warns:
        confidence = max(r["confidence"] for r in warns)
        # 轻微异常统一先转人工，避免把边界情况直接放过。
        decision = "review" if confidence < 0.98 else "pass"
    else:
        confidence = min([r.get("confidence", 0.8) for r in results.values()] or [0.8])
        decision = "pass"

    reasons: List[str] = []
    for name, r in results.items():
        if r["status"] in ("fail", "warn", "unknown") and r.get("reason"):
            reasons.append(f"{name}: {r['reason']}")

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "reasons": reasons,
    }


class AnalysisEngine:
    """组合质量、旋转、基准图、SOP 四层判定的引擎。"""

    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db

    def analyze(self, image_path: Path, camera: Dict[str, Any]) -> Dict[str, Any]:
        img = _read_image(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = self.config.thresholds

        quality = _quality(gray, t)
        rotation = _rotation(gray, t)
        metrics = dict(quality["metrics"])

        baseline_row = self.db.get_latest_baseline(camera["id"])
        baseline_path = Path(baseline_row["path"]) if baseline_row else None
        baseline, baseline_id = _baseline(img, baseline_path, t)
        if baseline_row:
            baseline_id = baseline_row["id"]

        template = camera.get("scene_template") or "default"
        sop_rules = self.config.sop_rules.get(template, self.config.sop_rules["default"])
        sop = _sop(metrics, template, sop_rules)

        module_results = {
            "quality": {k: quality[k] for k in ("status", "confidence", "reason")},
            "rotation": {k: rotation[k] for k in ("status", "confidence", "reason", "angle_deg")},
            "baseline": baseline,
            "sop": sop,
        }
        final = _fusion(module_results, t)
        final["metrics"] = metrics
        final["module_results"] = module_results
        final["baseline_id"] = baseline_id
        return final
