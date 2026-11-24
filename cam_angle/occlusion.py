"""Single-frame black/gray low-texture coverage detection, without a baseline.

Coverage is measured from pixel masks, never bounding-box areas. This is a
business image-quality rule, not semantic proof that an object covers the lens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


MAX_ANALYSIS_SIDE = 960
MAX_EDGE_DENSITY = 0.06
MIN_FLAT_FRACTION = 0.55
MAX_INTERIOR_EDGE_DENSITY = 0.045


def _letterbox_mask(
    hsv: np.ndarray, dark_value: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """识别播放器/编码器补出的纯黑边，避免把黑边算成镜头遮挡。

    只接受左右或上下成对出现、厚度不超过单边 20% 的近乎恒定黑条。
    单侧黑条和较宽黑块仍按真实遮挡处理，避免把遮住镜头的黑布误当黑边。
    """
    h, w = hsv.shape[:2]
    value = hsv[:, :, 2].astype(np.float32)
    bar_limit = min(10.0, max(4.0, dark_value * 0.35))
    near_black = value <= bar_limit
    max_rows = max(1, int(h * 0.20))
    max_cols = max(1, int(w * 0.20))

    row_fraction = near_black.mean(axis=1)
    row_std = value.std(axis=1)
    row_ok = (row_fraction >= 0.995) & (row_std <= 1.5)

    def edge_run(ok: np.ndarray, limit: int) -> Tuple[int, bool]:
        limit = min(limit, len(ok))
        run = 0
        while run < limit and ok[run]:
            run += 1
        too_thick = bool(run == limit and run < len(ok) and ok[run])
        return run, too_thick

    top, top_too_thick = edge_run(row_ok, max_rows)
    bottom, bottom_too_thick = edge_run(row_ok[::-1], max_rows)

    col_fraction = near_black.mean(axis=0)
    col_std = value.std(axis=0)
    col_ok = (col_fraction >= 0.995) & (col_std <= 1.5)
    left, left_too_thick = edge_run(col_ok, max_cols)
    right, right_too_thick = edge_run(col_ok[::-1], max_cols)

    mask = np.zeros((h, w), dtype=np.uint8)
    if top > 0 and bottom > 0 and not (top_too_thick or bottom_too_thick):
        mask[:top, :] = 1
        mask[h - bottom :, :] = 1
    else:
        top = bottom = 0
    if left > 0 and right > 0 and not (left_too_thick or right_too_thick):
        mask[:, :left] = 1
        mask[:, w - right :] = 1
    else:
        left = right = 0
    return mask, {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "ratio": round(float(mask.mean()), 6),
    }


def analyze_occlusion(img: np.ndarray, thresholds: Dict[str, float]) -> Tuple[Dict[str, Any], np.ndarray]:
    if img is None or img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("遮挡分析需要非空 BGR 图片")
    height, width = img.shape[:2]
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(height, width))
    work = cv2.resize(img, (max(1, round(width * scale)), max(1, round(height * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else img
    h, w = work.shape[:2]
    t = thresholds
    threshold = float(t.get("occlusion_area_ratio", 0.25))
    dark_value = float(t.get("occlusion_dark_value", 32))
    gray_saturation = float(t.get("occlusion_gray_saturation", 35))
    gray_max = float(t.get("occlusion_gray_max_value", 220))
    texture_limit = float(t.get("occlusion_texture_std", 10))
    min_region = float(t.get("occlusion_min_region_ratio", 0.005))

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    dark = hsv[:, :, 2] <= dark_value
    neutral = (hsv[:, :, 1] <= gray_saturation) & (hsv[:, :, 2] <= gray_max) & ~dark
    bars, bar_metrics = _letterbox_mask(hsv, dark_value)
    content = bars == 0
    content_area = int(np.count_nonzero(content)) or 1
    dark &= content
    neutral &= content
    color_mask = dark | neutral
    # Denoise sensor/JPEG noise without smoothing away ordinary scene structure.
    smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    window = max(5, min(21, int(min(h, w) * 0.025) | 1))
    mean = cv2.boxFilter(smooth, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    variance = cv2.boxFilter(smooth * smooth, -1, (window, window), borderType=cv2.BORDER_REFLECT) - mean * mean
    local_std = np.sqrt(np.maximum(variance, 0))
    edges = cv2.Canny(gray, 50, 130)
    edge_density = cv2.boxFilter((edges > 0).astype(np.float32), -1, (window, window))
    seeds = (color_mask & (local_std <= texture_limit) & (edge_density <= MAX_EDGE_DENSITY)).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Recover the boundary lost by the local window, but only inside actual
    # black/gray pixels. No filling bounding boxes or merging distant islands.
    expanded = cv2.dilate(seeds, np.ones((window, window), np.uint8))
    candidate = ((expanded > 0) & color_mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    accepted = np.zeros((h, w), np.uint8)
    regions = []
    discarded_texture = 0
    for label in range(1, count):
        x, y, rw, rh, area = (int(v) for v in stats[label])
        if area / content_area < min_region:
            continue
        component = labels == label
        interior = cv2.erode(component.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        interior_edges = float((edges[interior] > 0).mean()) if interior.any() else 0.0
        flat_fraction = float((local_std[component] <= texture_limit).mean())
        # A monochrome but detailed kitchen/night image must not become an
        # obstruction simply because its saturation is low.
        if flat_fraction < MIN_FLAT_FRACTION or interior_edges > MAX_INTERIOR_EDGE_DENSITY:
            discarded_texture += area
            continue
        accepted[component] = 255
        black_fraction = float(dark[component].mean())
        regions.append({
            "x": round(x / w, 6), "y": round(y / h, 6),
            "width": round(rw / w, 6), "height": round(rh / h, 6),
            "area_ratio": round(area / content_area, 6),
            "kind": "black" if black_fraction >= 0.8 else "gray" if black_fraction <= 0.2 else "mixed",
        })
    regions.sort(key=lambda r: r["area_ratio"], reverse=True)
    mask = cv2.resize(accepted, (width, height), interpolation=cv2.INTER_NEAREST) if scale < 1 else accepted
    ratio = float(np.count_nonzero(accepted > 0) / content_area)
    black_ratio = float(np.count_nonzero((accepted > 0) & dark) / content_area)
    gray_ratio = float(np.count_nonzero((accepted > 0) & neutral) / content_area)
    metrics = {
        "candidate_ratio": round(float(np.count_nonzero(candidate) / content_area), 6),
        "occlusion_ratio": round(ratio, 6),
        "black_region_ratio": round(black_ratio, 6),
        "gray_region_ratio": round(gray_ratio, 6),
        "largest_region_ratio": regions[0]["area_ratio"] if regions else 0.0,
        "raw_dark_ratio": round(float(np.count_nonzero(dark) / content_area), 6),
        "raw_gray_ratio": round(float(np.count_nonzero(neutral) / content_area), 6),
        "discarded_texture_ratio": round(discarded_texture / content_area, 6),
        "letterbox_ratio": bar_metrics["ratio"],
        "letterbox_edges": {
            "top": bar_metrics["top"],
            "bottom": bar_metrics["bottom"],
            "left": bar_metrics["left"],
            "right": bar_metrics["right"],
        },
        "region_count": len(regions), "threshold": threshold,
        "baseline_state": "disabled", "analysis_width": w, "analysis_height": h,
    }
    if min(height, width) < 64:
        status, confidence = "unknown", 0.0
        reason = "抓图尺寸过小，无法可靠判断黑灰遮挡区域"
    elif ratio > threshold:
        status, confidence = "fail", 0.95 if black_ratio > threshold else 0.88
        reason = f"黑灰低纹理区域占 {ratio:.1%}，超过 {threshold:.1%} 阈值（黑色 {black_ratio:.1%}，灰色 {gray_ratio:.1%}）"
    elif ratio >= max(0.0, threshold - 0.02) and ratio > 0:
        status, confidence = "warn", 0.65
        reason = f"黑灰低纹理区域占 {ratio:.1%}，接近 {threshold:.1%} 阈值，需人工确认"
    else:
        status, confidence = "pass", 0.90
        reason = f"黑灰低纹理区域占 {ratio:.1%}，未超过 {threshold:.1%} 阈值"
    return {
        "status": status, "confidence": confidence, "reason": reason,
        "metrics": metrics, "regions": regions,
        "method": "single_frame_black_gray_texture_v1",
        "overlay_available": False,
    }, mask


def save_occlusion_overlay(mask: np.ndarray, image_path: Path) -> bool:
    """Persist a lossless transparent mask alongside the capture, including empty masks."""
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255
    overlay[:, :, 3] = np.where(mask > 0, 110, 0).astype(np.uint8)
    try:
        success, encoded = cv2.imencode(".png", overlay)
        if not success:
            return False
        image_path.with_suffix(".occlusion.png").write_bytes(encoded.tobytes())
    except (OSError, cv2.error):
        # Evidence export failure must not turn a valid capture into no_signal.
        return False
    return True
