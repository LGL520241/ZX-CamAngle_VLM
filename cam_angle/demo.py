from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .analysis import AnalysisEngine
from .config import Config
from .database import Database


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _draw_scene(size=(640, 480)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (210, 220, 230)
    # 地面
    cv2.rectangle(img, (0, int(h * 0.65)), (w, h), (150, 150, 150), -1)
    # 货架
    cv2.rectangle(img, (80, 170), (260, 330), (80, 90, 110), -1)
    cv2.rectangle(img, (80, 170), (260, 330), (40, 45, 60), 3)
    for y in range(200, 320, 45):
        cv2.line(img, (80, y), (260, y), (220, 220, 220), 2)
    # 门口
    cv2.rectangle(img, (390, 90), (590, 360), (120, 70, 40), -1)
    cv2.rectangle(img, (390, 90), (590, 360), (30, 25, 20), 6)
    cv2.rectangle(img, (410, 110), (570, 340), (210, 200, 170), 3)
    # 顶部文字
    cv2.putText(img, "STORE VIEW", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    # 一些垂直线，方便旋转检测
    for x in (80, 260, 390, 590):
        cv2.line(img, (x, 90), (x, int(h * 0.65)), (40, 40, 40), 2)
    cv2.line(img, (0, int(h * 0.65)), (w, int(h * 0.65)), (40, 40, 40), 3)
    return img


def _make_image(kind: str, base: np.ndarray) -> np.ndarray:
    if kind == "normal":
        return base.copy()
    if kind == "black":
        return np.zeros_like(base)
    if kind == "white":
        img = np.zeros_like(base)
        img[:] = (245, 245, 245)
        return img
    if kind == "ceiling":
        img = np.zeros_like(base)
        img[:] = (188, 188, 188)
        noise = np.random.default_rng(7).normal(0, 2.0, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(base, (31, 31), 20)
    if kind == "rotated":
        (h, w) = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
        return cv2.warpAffine(base, M, (w, h), borderValue=(210, 220, 230))
    if kind == "slight_roll":
        (h, w) = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
        return cv2.warpAffine(base, M, (w, h), borderValue=(210, 220, 230))
    if kind == "moved":
        img = np.zeros_like(base)
        img[:] = (205, 215, 225)
        cv2.rectangle(img, (60, 210), (620, 440), (140, 150, 150), -1)
        cv2.rectangle(img, (130, 100), (360, 260), (70, 80, 100), -1)
        cv2.putText(img, "MOVED VIEW", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
        for x in (130, 360):
            cv2.line(img, (x, 100), (x, 440), (40, 40, 40), 3)
        return img
    return base.copy()


def _save_thumb(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb = cv2.resize(img, (320, 240))
    cv2.imwrite(str(path), thumb)


def seed_demo(db: Database, config: Config) -> Dict[str, Any]:
    """在空库中生成演示数据；已存在相机时跳过，避免覆盖真实数据。"""
    if db.list_cameras():
        return {"seeded": False, "reason": "已有相机数据，跳过演示数据生成"}

    engine = AnalysisEngine(config, db)
    base_scene = _draw_scene()
    stores = [("A001", "indoor"), ("A002", "outdoor"), ("A003", "default")]
    specs: List[Dict[str, Any]] = [
        {"kind": "normal", "store": 0},
        {"kind": "black", "store": 0},
        {"kind": "ceiling", "store": 0},
        {"kind": "blur", "store": 0},
        {"kind": "rotated", "store": 1},
        {"kind": "moved", "store": 1},
        {"kind": "normal", "store": 1, "baseline": False},
        {"kind": "slight_roll", "store": 1},
        {"kind": "white", "store": 2},
        {"kind": "normal", "store": 2},
        {"kind": "moved", "store": 2},
        {"kind": "black", "store": 2},
    ]

    created_cameras = 0
    created_captures = 0
    for idx, spec in enumerate(specs, start=1):
        store_id, template = stores[spec["store"]]
        device_serial = f"DEMO{1000 + idx}"
        channel_no = 1
        camera_id = db.upsert_camera(
            device_serial=device_serial,
            channel_no=channel_no,
            channel_name=f"演示相机 {idx}",
            group_no="1",
            store_id=store_id,
            scene_template=template,
            status="active",
            meta={"demo": True, "kind": spec["kind"]},
        )
        created_cameras += 1

        has_baseline = spec.get("baseline", True)
        baseline_id = None
        if has_baseline:
            baseline_path = config.baselines_dir / f"{device_serial}_{channel_no}_day.jpg"
            cv2.imwrite(str(baseline_path), base_scene)
            baseline_id = db.add_baseline(camera_id, str(baseline_path), kind="day", meta={"demo": True})

        img = _make_image(spec["kind"], base_scene)
        image_path = config.images_dir / f"{device_serial}_{channel_no}.jpg"
        thumb_path = config.thumbs_dir / f"{device_serial}_{channel_no}.jpg"
        cv2.imwrite(str(image_path), img)
        _save_thumb(img, thumb_path)

        camera = db.get_camera(camera_id)
        result = engine.analyze(image_path, camera)
        capture_id = db.add_capture(
            camera_id=camera_id,
            path=str(image_path),
            thumb_path=str(thumb_path),
            source="demo",
            quality=0,
            capture_time=_now(),
            decision=result["decision"],
            confidence=result["confidence"],
            reasons=result["reasons"],
            metrics=result["metrics"],
            module_results=result["module_results"],
            baseline_id=baseline_id,
        )
        db.update_capture_result(
            capture_id,
            result["decision"],
            result["confidence"],
            result["reasons"],
            result["metrics"],
            result["module_results"],
            baseline_id,
        )
        created_captures += 1

    return {"seeded": True, "cameras": created_cameras, "captures": created_captures}


def render_demo_image(kind: str) -> np.ndarray:
    return _make_image(kind, _draw_scene())
