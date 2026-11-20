from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .analysis import AnalysisEngine
from .config import Config
from .database import Database
from .demo import render_demo_image, seed_demo
from .hikvision import HikvisionClient


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _save_thumb(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb = cv2.resize(img, (320, 240))
    cv2.imwrite(str(path), thumb)


class Pipeline:
    """负责相机发现、抓图调度、图片落盘、分析入库。"""

    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db
        self.engine = AnalysisEngine(config, db)
        self.hik = HikvisionClient(config.hik_base_url, config.hik_client_id, config.hik_client_secret)
        self._semaphore = asyncio.Semaphore(max(1, config.concurrency))
        self._last_capture: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._active_tasks: Dict[int, Dict[str, Any]] = {}
        self._cycle_counts: Dict[str, int] = {"pass": 0, "fail": 0, "review": 0, "no_signal": 0}
        self._cycle_start_mono: float = 0.0

    @staticmethod
    def _camera_meta(camera: Dict[str, Any]) -> Dict[str, Any]:
        meta = camera.get("meta")
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except Exception:
                return {}
        return meta or {}

    def _set_progress(self, camera: Dict[str, Any], step: str) -> None:
        self._active_tasks[camera["id"]] = {
            "camera_id": camera["id"],
            "device_serial": camera["device_serial"],
            "channel_no": camera["channel_no"],
            "channel_name": camera.get("channel_name") or "",
            "step": step,
            "started_at": _now(),
        }

    def _finish_progress(self, camera_id: int, decision: Optional[str] = None) -> None:
        self._active_tasks.pop(camera_id, None)
        if decision in self._cycle_counts:
            self._cycle_counts[decision] += 1

    def progress_snapshot(self) -> Dict[str, Any]:
        processed = int(self.db.get_cycle("processed", 0))
        total = int(self.db.get_cycle("total", 0))
        percent = round(processed * 100 / total, 1) if total else 0.0
        elapsed = time.monotonic() - self._cycle_start_mono if self._cycle_start_mono else 0.0
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta = (total - processed) / rate if rate > 0 else 0.0
        return {
            "state": self.db.get_cycle("state", "idle"),
            "total": total,
            "processed": processed,
            "percent": percent,
            "started_at": self.db.get_cycle("started_at", ""),
            "finished_at": self.db.get_cycle("finished_at", ""),
            "error": self.db.get_cycle("error", ""),
            "active": list(self._active_tasks.values()),
            "by_decision": dict(self._cycle_counts),
            "rate_per_sec": round(rate, 2),
            "eta_seconds": int(eta),
        }

    async def ensure_cameras(self) -> int:
        if self.config.demo_mode:
            if not self.db.list_cameras():
                seed_demo(self.db, self.config)
            return len(self.db.list_cameras())

        group_nos = [g.strip() for g in self.config.hik_group_nos.split(",") if g.strip()]
        targets = await self.hik.discover_cameras(group_nos, self.config.region_filter)
        # SQLite 是同步操作，放到线程里执行，避免几千条入库阻塞事件循环。
        def _sync_targets() -> None:
            for target in targets:
                self.db.upsert_camera(
                    device_serial=target["device_serial"],
                    channel_no=target["channel_no"],
                    channel_name=target.get("channel_name", ""),
                    group_no=target.get("group_no", ""),
                    store_id=target.get("store_id", ""),
                    scene_template="default",
                    status="active",
                    meta={
                        "store_name": target.get("store_name", ""),
                        "store_no": target.get("store_no", ""),
                    },
                )

        await asyncio.to_thread(_sync_targets)
        return len(targets)

    async def _rate_limit(self, device_serial: str, channel_no: int) -> None:
        key = f"{device_serial}:{channel_no}"
        while True:
            async with self._lock:
                last = self._last_capture.get(key, 0.0)
                now = time.monotonic()
                wait = self.config.capture_interval - (now - last)
                if wait <= 0:
                    self._last_capture[key] = now
                    return
            await asyncio.sleep(min(wait, self.config.capture_interval))

    async def _process_camera(self, camera: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        camera_id = camera["id"]
        serial = camera["device_serial"]
        channel_no = camera["channel_no"]
        device_key = f"{serial}_{channel_no}"
        self._set_progress(camera, "抓图")
        result: Optional[Dict[str, Any]] = None

        try:
            if self.config.demo_mode:
                self._set_progress(camera, "生成演示图")
                camera_row = self.db.get_camera(camera_id)
                meta_raw = (camera_row or {}).get("meta", "{}")
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                kind = (meta or {}).get("kind", "normal")
                img = render_demo_image(kind)
                image_path = self.config.images_dir / f"{device_key}_{int(time.time())}.jpg"
                cv2.imwrite(str(image_path), img)
                source = "demo"
                quality = 0
            else:
                await self._rate_limit(serial, channel_no)
                url = await self.hik.capture(serial, channel_no, quality=0)
                self._set_progress(camera, "下载")
                image_path = self.config.images_dir / f"{device_key}_{int(time.time())}.jpg"
                await self.hik.download(url, image_path)
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("下载图片解码失败")
                source = "capture"
                quality = 0

            self._set_progress(camera, "分析")
            thumb_path = self.config.thumbs_dir / f"{device_key}_{int(time.time())}.jpg"
            _save_thumb(img, thumb_path)
            camera_row = self.db.get_camera(camera_id)
            analysis = self.engine.analyze(image_path, camera_row)
            baseline_id = analysis.get("baseline_id")
            self._set_progress(camera, "入库")
            capture_id = self.db.add_capture(
                camera_id=camera_id,
                path=str(image_path),
                thumb_path=str(thumb_path),
                source=source,
                quality=quality,
                capture_time=_now(),
                decision=analysis["decision"],
                confidence=analysis["confidence"],
                reasons=analysis["reasons"],
                metrics=analysis["metrics"],
                module_results=analysis["module_results"],
                baseline_id=baseline_id,
            )
            self.db.update_capture_result(
                capture_id,
                analysis["decision"],
                analysis["confidence"],
                analysis["reasons"],
                analysis["metrics"],
                analysis["module_results"],
                baseline_id,
            )
            result = {"capture_id": capture_id, **analysis}
        except Exception as exc:  # 抓图失败也要留痕
            image_path = self.config.images_dir / f"{device_key}_{int(time.time())}_failed.jpg"
            capture_id = self.db.add_capture(
                camera_id=camera_id,
                path="",
                thumb_path="",
                source="capture",
                quality=0,
                capture_time=_now(),
                decision="no_signal",
                confidence=0.0,
                reasons=[f"抓图失败: {exc}"],
                metrics={},
                module_results={},
                baseline_id=None,
            )
            result = {
                "capture_id": capture_id,
                "decision": "no_signal",
                "confidence": 0.0,
                "reasons": [str(exc)],
            }
        finally:
            self._finish_progress(camera_id, result.get("decision") if result else None)
        return result

    async def run_cycle(self) -> Dict[str, Any]:
        if self._running:
            return {"status": "already_running"}
        self._running = True
        self._cycle_start_mono = time.monotonic()
        self._active_tasks.clear()
        for key in self._cycle_counts:
            self._cycle_counts[key] = 0
        self.db.set_cycle("state", "running")
        self.db.set_cycle("started_at", _now())
        self.db.set_cycle("processed", 0)
        try:
            # 启动时已经同步过设备；已有相机列表就直接使用，避免每轮巡检重新全量拉取。
            if self.config.demo_mode or not self.db.list_cameras("active"):
                await self.ensure_cameras()
            cameras = self.db.list_cameras("active")
            if not self.config.demo_mode:
                cameras = [
                    c
                    for c in cameras
                    if not self._camera_meta(c).get("demo")
                ]
            if self.config.region_filter:
                cameras = [
                    c
                    for c in cameras
                    if self.config.region_filter in self._camera_meta(c).get("store_name", "")
                ]
            if self.config.sample_size > 0:
                cameras = random.sample(cameras, min(self.config.sample_size, len(cameras)))
            total = len(cameras)
            self.db.set_cycle("total", total)
            sem = self._semaphore

            async def worker(cam: Dict[str, Any]) -> Dict[str, Any]:
                async with sem:
                    return await self._process_camera(cam)

            processed = 0
            for fut in asyncio.as_completed([worker(c) for c in cameras]):
                await fut
                processed += 1
                self.db.set_cycle("processed", processed)
                self.db.set_cycle("state", "running")

            self.db.set_cycle("state", "finished")
            self.db.set_cycle("finished_at", _now())
            return {"status": "finished", "total": total, "processed": processed}
        except Exception as exc:
            self.db.set_cycle("state", "error")
            self.db.set_cycle("error", str(exc))
            return {"status": "error", "error": str(exc)}
        finally:
            self._running = False
