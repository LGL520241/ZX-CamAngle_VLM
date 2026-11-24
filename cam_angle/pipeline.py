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
from .artifacts import ArtifactStore
from .blur import analyze_blur
from .config import Config
from .database import Database
from .decision import fuse_semantic_decision
from .demo import render_demo_image, seed_demo
from .hikvision import HikvisionClient
from .occlusion import analyze_occlusion
from .vlm import QwenSemanticClient


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _save_thumb(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = img.shape[:2]
    scale = min(320.0 / max(width, 1), 240.0 / max(height, 1))
    thumb = cv2.resize(
        img,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        path.write_bytes(encoded.tobytes())


def _semantic_quality_guard(image_path: Path, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """仅用于复核 Qwen 的 severe 结论，不作为常态化第三路检出。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"available": False, "reason": "图片解码失败"}
    occlusion, occlusion_mask = analyze_occlusion(image, thresholds)
    blur = analyze_blur(image, thresholds, exclusion_mask=occlusion_mask)
    height, width = image.shape[:2]
    central = image[
        int(height * 0.05):int(height * 0.95),
        int(width * 0.15):int(width * 0.85),
    ]
    central_occlusion, central_mask = analyze_occlusion(central, thresholds)
    central_blur = analyze_blur(
        central, thresholds, exclusion_mask=central_mask
    )
    return {
        "available": True,
        "occlusion": {
            "status": occlusion.get("status"),
            "ratio": (occlusion.get("metrics") or {}).get("occlusion_ratio", 0.0),
            "reason": occlusion.get("reason"),
        },
        "blur": {
            "status": blur.get("status"),
            "score": (blur.get("metrics") or {}).get("blur_score", 0.0),
            "affected_ratio": (blur.get("metrics") or {}).get("blur_affected_ratio", 0.0),
            "reason": blur.get("reason"),
        },
        "central": {
            "occlusion": {
                "status": central_occlusion.get("status"),
                "ratio": (central_occlusion.get("metrics") or {}).get(
                    "occlusion_ratio", 0.0
                ),
                "reason": central_occlusion.get("reason"),
            },
            "blur": {
                "status": central_blur.get("status"),
                "score": (central_blur.get("metrics") or {}).get("blur_score", 0.0),
                "affected_ratio": (central_blur.get("metrics") or {}).get(
                    "blur_affected_ratio", 0.0
                ),
                "reason": central_blur.get("reason"),
            },
        },
    }


class Pipeline:
    """负责相机发现、抓图调度、图片落盘、分析入库。"""

    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db
        self.engine = AnalysisEngine(config, db)
        self.hik = HikvisionClient(config.hik_base_url, config.hik_client_id, config.hik_client_secret)
        self.vlm = QwenSemanticClient(config)
        self.artifacts = ArtifactStore(config.artifact_root or config.data_dir / "artifacts")
        self._last_capture: Dict[str, float] = {}
        # 在运行中的事件循环内创建锁，兼容 Python 3.9 的事件循环绑定。
        self._lock: Optional[asyncio.Lock] = None
        self._running = False
        self._active_tasks: Dict[int, Dict[str, Any]] = {}
        self._cycle_counts: Dict[str, int] = {"pass": 0, "fail": 0, "review": 0, "no_signal": 0}
        self._cycle_start_mono: float = 0.0
        self._batch_id: Optional[int] = None
        self._cycle_dir: Optional[Path] = None
        self._cycle_pipeline_mode: str = config.pipeline_mode
        self._cycle_module_switches: Dict[str, bool] = dict(config.module_switches)
        self._cycle_vlm_prompts: Dict[str, str] = dict(config.vlm_prompts)

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
            "batch_id": self._batch_id or self.db.get_cycle("batch_id", None),
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
        if self._lock is None:
            self._lock = asyncio.Lock()
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

    def _record_capture_error(
        self,
        camera: Dict[str, Any],
        capture_started_at: str,
        paths: Dict[str, Path],
        exc: Exception,
    ) -> Dict[str, Any]:
        capture_id = self.db.add_capture(
            camera_id=camera["id"],
            path="",
            thumb_path="",
            source="capture",
            quality=0,
            capture_time=capture_started_at,
            decision="no_signal",
            confidence=0.0,
            reasons=[f"抓图失败: {exc}"],
            metrics={},
            module_results={},
            baseline_id=None,
            batch_id=self._batch_id,
        )
        result = {
            "capture_id": capture_id,
            "decision": "no_signal",
            "confidence": 0.0,
            "reasons": [str(exc)],
        }
        self.artifacts.write_json(paths["error"], {
            **result,
            "capture_time": capture_started_at,
            "device_serial": camera["device_serial"],
            "channel_no": camera["channel_no"],
            "error_type": type(exc).__name__,
        })
        self.artifacts.append_event(self._cycle_dir, {
            "phase": "capture_error",
            "capture_id": capture_id,
            "capture_time": capture_started_at,
            "device_serial": camera["device_serial"],
            "channel_no": camera["channel_no"],
            "decision": "no_signal",
            "confidence": 0.0,
            "reasons": [str(exc)],
        })
        return result

    async def _capture_camera_only(self, camera: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解耦链路第一阶段：只抓图、下载和解码，不执行推理。"""
        if self._cycle_dir is None:
            raise RuntimeError("巡检产物目录尚未初始化")
        capture_started_at = _now()
        paths = self.artifacts.capture_paths(
            self._cycle_dir,
            capture_started_at,
            camera["device_serial"],
            camera["channel_no"],
        )
        image_path = paths["source"]
        thumb_path = paths["thumb"]
        self._set_progress(camera, "抓图")
        try:
            if self.config.demo_mode:
                self._set_progress(camera, "生成演示图")
                camera_row = self.db.get_camera(camera["id"])
                meta_raw = (camera_row or {}).get("meta", "{}")
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                img = render_demo_image((meta or {}).get("kind", "normal"))
                ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("演示图片编码失败")
                image_path.write_bytes(encoded.tobytes())
                source = "demo"
            else:
                await self._rate_limit(camera["device_serial"], camera["channel_no"])
                url = await self.hik.capture(camera["device_serial"], camera["channel_no"], quality=0)
                self._set_progress(camera, "下载")
                await self.hik.download(url, image_path)
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("下载图片解码失败")
                source = "capture"
            _save_thumb(img, thumb_path)
            return {
                "camera": camera,
                "capture_time": capture_started_at,
                "paths": paths,
                "source": source,
                "quality": 0,
            }
        except Exception as exc:
            result = self._record_capture_error(camera, capture_started_at, paths, exc)
            self._finish_progress(camera["id"], result["decision"])
            return None

    async def _process_camera(
        self,
        camera: Dict[str, Any],
        captured: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        camera_id = camera["id"]
        serial = camera["device_serial"]
        channel_no = camera["channel_no"]
        if self._cycle_dir is None:
            raise RuntimeError("巡检产物目录尚未初始化")
        capture_started_at = captured["capture_time"] if captured else _now()
        paths = captured["paths"] if captured else self.artifacts.capture_paths(
            self._cycle_dir, capture_started_at, serial, channel_no
        )
        image_path = paths["source"]
        thumb_path = paths["thumb"]
        if captured is None:
            self._set_progress(camera, "抓图")
        result: Optional[Dict[str, Any]] = None

        try:
            if captured is not None:
                source = captured["source"]
                quality = captured["quality"]
            elif self.config.demo_mode:
                self._set_progress(camera, "生成演示图")
                camera_row = self.db.get_camera(camera_id)
                meta_raw = (camera_row or {}).get("meta", "{}")
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                kind = (meta or {}).get("kind", "normal")
                img = render_demo_image(kind)
                ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError("演示图片编码失败")
                image_path.write_bytes(encoded.tobytes())
                source = "demo"
                quality = 0
            else:
                await self._rate_limit(serial, channel_no)
                url = await self.hik.capture(serial, channel_no, quality=0)
                self._set_progress(camera, "下载")
                await self.hik.download(url, image_path)
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("下载图片解码失败")
                source = "capture"
                quality = 0

            self._set_progress(camera, "分析")
            if captured is None:
                _save_thumb(img, thumb_path)
            camera_row = self.db.get_camera(camera_id) or camera
            # 图像分析包含多次 OpenCV 运算，不能阻塞巡检进度和网页 API。
            cv_analysis = await asyncio.to_thread(
                self.engine.analyze,
                image_path,
                camera_row,
                dict(self._cycle_module_switches),
            )
            self.artifacts.write_json(paths["cv"], cv_analysis)
            self.artifacts.append_event(self._cycle_dir, {
                "phase": "cv_complete",
                "capture_time": capture_started_at,
                "device_serial": serial,
                "channel_no": channel_no,
                "image_path": str(image_path),
                "decision": cv_analysis.get("decision"),
                "module_results": cv_analysis.get("module_results", {}),
            })

            self._set_progress(camera, "语义分析")
            qwen_enabled = bool(
                self._cycle_module_switches.get("qwen_semantic", True)
                and self.config.vlm_enabled
            )
            if qwen_enabled:
                semantic = await self.vlm.analyze(
                    image_path,
                    camera_row,
                    prompts=dict(self._cycle_vlm_prompts),
                )
            else:
                semantic = {
                    "status": "skipped",
                    "confidence": 0.0,
                    "reason": (
                        "Qwen 语义模块已关闭"
                        if not self._cycle_module_switches.get("qwen_semantic", True)
                        else "Qwen 服务未启用"
                    ),
                    "data": {},
                    "metrics": {},
                }
            semantic_data = semantic.get("data") or {}
            if qwen_enabled and (
                semantic_data.get("occlusion") == "severe"
                or semantic_data.get("blur") == "severe"
            ):
                guard = await asyncio.to_thread(
                    _semantic_quality_guard,
                    image_path,
                    dict(self.config.thresholds),
                )
                cv_analysis.setdefault("module_results", {})["quality_guard"] = guard
                self.artifacts.write_json(paths["quality_guard"], guard)
            request_log = dict(semantic.get("request") or {})
            prepared_bytes = semantic.pop("_prepared_image", None)
            if prepared_bytes:
                vlm_input = paths["vlm_input"]
                vlm_input.write_bytes(prepared_bytes)
                request_log["vlm_input_path"] = str(vlm_input)
            request_log["capture_time"] = capture_started_at
            request_log["camera"] = {
                "device_serial": serial,
                "channel_no": channel_no,
                "channel_name": camera.get("channel_name", ""),
            }
            self.artifacts.write_json(paths["vlm_request"], request_log)
            self.artifacts.write_json(paths["vlm_response"], {
                "status": semantic.get("status"),
                "confidence": semantic.get("confidence"),
                "reason": semantic.get("reason"),
                "data": semantic.get("data"),
                "metrics": semantic.get("metrics"),
                "obstruction_verification": semantic.get("obstruction_verification"),
                "operation_verification": semantic.get("operation_verification"),
                "raw_response": semantic.get("raw_response", ""),
                "error": semantic.get("error", ""),
            })
            self.artifacts.append_event(self._cycle_dir, {
                "phase": "vlm_complete",
                "capture_time": capture_started_at,
                "device_serial": serial,
                "channel_no": channel_no,
                "status": semantic.get("status"),
                "confidence": semantic.get("confidence"),
                "metrics": semantic.get("metrics", {}),
                "obstruction_verification": semantic.get("obstruction_verification"),
                "operation_verification": semantic.get("operation_verification"),
                "data": semantic.get("data", {}),
            })
            if qwen_enabled:
                analysis = fuse_semantic_decision(cv_analysis, semantic, self.config)
            else:
                analysis = cv_analysis
            self.artifacts.write_json(paths["analysis"], analysis)
            baseline_id = analysis.get("baseline_id")
            self._set_progress(camera, "入库")
            capture_id = self.db.add_capture(
                camera_id=camera_id,
                path=str(image_path),
                thumb_path=str(thumb_path),
                source=source,
                quality=quality,
                capture_time=capture_started_at,
                decision=analysis["decision"],
                confidence=analysis["confidence"],
                reasons=analysis["reasons"],
                metrics=analysis["metrics"],
                module_results=analysis["module_results"],
                baseline_id=baseline_id,
                batch_id=self._batch_id,
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
            self.artifacts.append_event(self._cycle_dir, {
                "phase": "capture_complete",
                "capture_id": capture_id,
                "capture_time": capture_started_at,
                "device_serial": serial,
                "channel_no": channel_no,
                "decision": analysis["decision"],
                "confidence": analysis["confidence"],
                "reasons": analysis["reasons"],
                "artifact_dir": str(self._cycle_dir),
            })
        except Exception as exc:  # 抓图失败也要留痕
            result = self._record_capture_error(
                camera, capture_started_at, paths, exc
            )
        finally:
            self._finish_progress(camera_id, result.get("decision") if result else None)
        return result

    async def run_cycle(self) -> Dict[str, Any]:
        if self._running:
            return {"status": "already_running"}
        self._running = True
        self._cycle_start_mono = time.monotonic()
        self._active_tasks.clear()
        started_at = _now()
        batch_id = self.db.create_batch(started_at)
        self._batch_id = batch_id
        self._cycle_dir = self.artifacts.begin_cycle(started_at, batch_id)
        self._cycle_pipeline_mode = self.config.pipeline_mode
        self._cycle_module_switches = dict(self.config.module_switches)
        self._cycle_vlm_prompts = dict(self.config.vlm_prompts)
        self.artifacts.write_json(self._cycle_dir / "cycle-start.json", {
            "batch_id": batch_id,
            "started_at": started_at,
            "artifact_dir": str(self._cycle_dir),
            "vlm_enabled": self.config.vlm_enabled,
            "vlm_model": self.config.vlm_model,
            "module_switches": dict(self._cycle_module_switches),
            "pipeline_mode": self._cycle_pipeline_mode,
            "concurrency": self.config.concurrency,
            "capture_concurrency": self.config.capture_concurrency,
            "inference_concurrency": self.config.inference_concurrency,
            "vlm_prompts": dict(self._cycle_vlm_prompts),
        })
        for key in self._cycle_counts:
            self._cycle_counts[key] = 0
        self.db.set_cycle("state", "running")
        self.db.set_cycle("batch_id", batch_id)
        self.db.set_cycle("started_at", started_at)
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
            self.db.update_batch(batch_id, total=total)
            self._lock = asyncio.Lock()
            processed = 0
            if self._cycle_pipeline_mode == "decoupled":
                capture_sem = asyncio.Semaphore(max(1, self.config.capture_concurrency))

                async def capture_worker(cam: Dict[str, Any]):
                    async with capture_sem:
                        return await self._capture_camera_only(cam)

                captured_items = []
                for fut in asyncio.as_completed([capture_worker(c) for c in cameras]):
                    captured = await fut
                    if captured is not None:
                        captured_items.append(captured)
                    else:
                        processed += 1
                        self.db.set_cycle("processed", processed)
                        self.db.update_batch(batch_id, processed=processed, state="running")

                inference_sem = asyncio.Semaphore(
                    max(1, self.config.inference_concurrency)
                )

                async def inference_worker(captured: Dict[str, Any]):
                    async with inference_sem:
                        return await self._process_camera(
                            captured["camera"], captured
                        )

                for fut in asyncio.as_completed(
                    [inference_worker(item) for item in captured_items]
                ):
                    await fut
                    processed += 1
                    self.db.set_cycle("processed", processed)
                    self.db.set_cycle("state", "running")
                    self.db.update_batch(batch_id, processed=processed, state="running")
            else:
                sem = asyncio.Semaphore(max(1, self.config.concurrency))

                async def worker(cam: Dict[str, Any]) -> Dict[str, Any]:
                    async with sem:
                        return await self._process_camera(cam)

                for fut in asyncio.as_completed([worker(c) for c in cameras]):
                    await fut
                    processed += 1
                    self.db.set_cycle("processed", processed)
                    self.db.set_cycle("state", "running")
                    self.db.update_batch(batch_id, processed=processed, state="running")

            self.db.set_cycle("state", "finished")
            finished_at = _now()
            self.db.set_cycle("finished_at", finished_at)
            self.db.update_batch(
                batch_id,
                state="finished",
                processed=processed,
                finished_at=finished_at,
            )
            summary = {
                "status": "finished",
                "batch_id": batch_id,
                "total": total,
                "processed": processed,
                "started_at": started_at,
                "finished_at": finished_at,
                "by_decision": dict(self._cycle_counts),
                "module_switches": dict(self._cycle_module_switches),
                "pipeline_mode": self._cycle_pipeline_mode,
                "capture_concurrency": self.config.capture_concurrency,
                "inference_concurrency": self.config.inference_concurrency,
                "artifact_dir": str(self._cycle_dir),
            }
            self.artifacts.write_cycle(self._cycle_dir, summary)
            self.artifacts.append_event(self._cycle_dir, {"phase": "cycle_complete", **summary})
            return summary
        except Exception as exc:
            self.db.set_cycle("state", "error")
            self.db.set_cycle("error", str(exc))
            finished_at = _now()
            self.db.set_cycle("finished_at", finished_at)
            self.db.update_batch(
                batch_id,
                state="error",
                finished_at=finished_at,
                error=str(exc),
            )
            summary = {
                "status": "error",
                "batch_id": batch_id,
                "error": str(exc),
                "artifact_dir": str(self._cycle_dir),
            }
            self.artifacts.write_cycle(self._cycle_dir, summary)
            self.artifacts.append_event(self._cycle_dir, {"phase": "cycle_error", **summary})
            return summary
        finally:
            self._running = False
