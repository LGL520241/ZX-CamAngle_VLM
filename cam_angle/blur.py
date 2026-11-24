"""可解释的分区清晰度检查；不把纯色表面或噪声当作失焦证据。"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def analyze_blur(
    img: np.ndarray,
    thresholds: Dict[str, float],
    baseline: Optional[np.ndarray] = None,
    exclusion_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """检查有场景结构的有效区域，返回 pass / fail / warn / unknown。

    最长边超过 640 时先降采样，避免不同抓图分辨率直接套用拉普拉斯
    阈值。不放大小图；通过中值滤波抑制传感器噪声，并将画面分成
    4×4 块，避免清楚的时间水印掩盖主体失焦。exclusion_mask 非零
    像素（遮挡区域）不参与检查。baseline 参数保留接口兼容性，未经
    几何验证的基准图不会用于宣称画面清晰度退化。
    """
    if img is None or img.ndim not in (2, 3) or min(img.shape[:2]) < 64:
        return {
            "status": "unknown", "confidence": 0.0,
            "reason": "图片尺寸过小，无法可靠判断清晰度", "metrics": {},
        }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h0, w0 = gray.shape
    valid = np.ones((h0, w0), dtype=np.uint8)
    if exclusion_mask is not None:
        if exclusion_mask.shape[:2] != gray.shape:
            raise ValueError("清晰度排除区域尺寸必须与原图一致")
        if exclusion_mask.ndim != 2:
            raise ValueError("清晰度排除区域必须为二维遮罩")
        valid[exclusion_mask != 0] = 0
    scale = min(1.0, 640.0 / max(h0, w0))
    if scale < 1.0:
        size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    h, w = gray.shape
    # 滤波邻域触及遮挡边界时，也应排除该像素的人工高频响应。
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    clean = cv2.medianBlur(gray, 3)
    raw_lap = cv2.Laplacian(gray, cv2.CV_32F)
    clean_lap = cv2.Laplacian(clean, cv2.CV_32F)
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    # 单一方向的运动模糊可能保留另一方向的锐利边缘，整图方差会漏检。
    # 方向性也可能来自货架等正常结构，因此只作为转人工的依据。
    lap_x = cv2.Sobel(clean, cv2.CV_32F, 2, 0, ksize=3)
    lap_y = cv2.Sobel(clean, cv2.CV_32F, 0, 2, ksize=3)
    threshold = float(thresholds.get("blur_var", 60.0))
    min_tiles = int(thresholds.get("blur_min_tiles", 4))
    fail_ratio = float(thresholds.get("blur_region_ratio", 0.5))
    samples = []
    covered = 0
    noisy_tiles = 0
    usable_tiles = 0
    tile_metrics = []
    directional_tiles = 0
    for row in range(4):
        for col in range(4):
            ys = slice(row * h // 4 + 3, (row + 1) * h // 4 - 3)
            xs = slice(col * w // 4 + 3, (col + 1) * w // 4 - 3)
            tile_valid = valid[ys, xs] != 0
            if not tile_valid.size or tile_valid.mean() < 0.6:
                continue
            usable_tiles += 1
            values = clean[ys, xs][tile_valid].astype(np.float32)
            raw_values = gray[ys, xs][tile_valid].astype(np.float32)
            grad_values = gradient[ys, xs][tile_valid]
            residual = raw_values - values
            noise_ratio = float(residual.std() / max(float(raw_values.std()), 1.0))
            if noise_ratio > 0.65 and raw_values.std() > 8.0:
                noisy_tiles += 1
                continue
            # 纯墙、亮度渐变无可辨结构，不能只因拉普拉斯小就判模糊。
            dynamic_range = float(np.percentile(values, 95) - np.percentile(values, 5))
            informative = bool(values.std() >= 6.0 and dynamic_range >= 18.0 and grad_values.std() >= 3.0)
            if not informative:
                continue
            score = float(clean_lap[ys, xs][tile_valid].var())
            vx = float(lap_x[ys, xs][tile_valid].var())
            vy = float(lap_y[ys, xs][tile_valid].var())
            direction_ratio = min(vx, vy) / max(vx, vy, 1e-6)
            if direction_ratio < 0.03 and max(vx, vy) > threshold:
                directional_tiles += 1
            covered += int(tile_valid.sum())
            samples.append(score)
            tile_metrics.append({
                "row": row, "column": col, "score": round(score, 2),
                "blurry": bool(score < threshold),
                "direction_ratio": round(direction_ratio, 4),
            })
    count = len(samples)
    blurry = sum(score < threshold for score in samples)
    ratio = blurry / count if count else 0.0
    valid_pixels = valid != 0
    raw_score = float(raw_lap[valid_pixels].var()) if valid_pixels.any() else 0.0
    score = float(np.median(samples)) if samples else 0.0
    noise_ratio = noisy_tiles / usable_tiles if usable_tiles else 0.0
    coverage = covered / max(h * w, 1)
    metrics = {
        "blur_score": round(score, 2),
        "blur_raw_score": round(raw_score, 2),
        "blur_affected_ratio": round(ratio, 4),
        "blur_informative_tiles": count,
        "blur_blurry_tiles": blurry,
        "blur_coverage": round(coverage, 4),
        "blur_noisy_tile_ratio": round(noise_ratio, 4),
        "blur_analysis_width": w, "blur_analysis_height": h,
        "blur_threshold": threshold,
        "blur_directional_tiles": directional_tiles,
        "blur_baseline_used": False,
        "blur_tiles": tile_metrics,
    }
    if noise_ratio >= 0.5:
        status, confidence, reason = "warn", 0.65, "画面噪声较强，不能将噪声当作清晰细节，需人工确认"
    elif count < min_tiles or coverage < 0.15:
        status, confidence, reason = "unknown", 0.0, "可辨场景结构不足或遮挡过多，无法可靠判断是否失焦"
    elif ratio >= fail_ratio:
        # 大面积结构都有明显退化时才自动判失败，少数边界块转人工。
        status, confidence = ("fail", 0.90) if coverage >= 0.35 else ("warn", 0.70)
        reason = f"主体画面疑似模糊：{blurry}/{count} 个有效分区清晰度不足（{ratio:.0%}）"
    elif ratio >= max(0.2, fail_ratio * 0.5):
        status, confidence = "warn", 0.70
        reason = f"部分画面疑似模糊：{blurry}/{count} 个有效分区清晰度不足，建议人工复核"
    elif directional_tiles >= min_tiles and directional_tiles / count >= 0.5:
        status, confidence = "warn", 0.65
        reason = "画面多个分区存在明显方向性细节损失，疑似运动模糊，需人工确认"
    else:
        status, confidence, reason = "pass", 0.93, "有效场景分区清晰度正常"
    return {"status": status, "confidence": confidence, "reason": reason, "metrics": metrics}
