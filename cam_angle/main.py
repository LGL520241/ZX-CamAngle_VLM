from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (
    get_config,
    restore_module_switches,
    restore_thresholds,
    validate_module_switches,
    validate_pipeline_mode,
    validate_thresholds,
)
from .database import Database
from .demo import seed_demo
from .pipeline import Pipeline
from .vlm import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


config = get_config()
db = Database(config.db_path)
pipeline = Pipeline(config, db)

app = FastAPI(title="CamAngle 摄像头角度巡检", version="0.1.0")


@app.middleware("http")
async def no_cache_for_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup() -> None:
    config.ensure_dirs()
    config.thresholds = restore_thresholds(config.thresholds, db.get_setting("thresholds", {}))
    config.module_switches = restore_module_switches(
        db.get_setting("module_switches", {})
    )
    try:
        config.pipeline_mode = validate_pipeline_mode(
            db.get_setting("pipeline_mode", config.pipeline_mode)
        )
    except ValueError:
        config.pipeline_mode = "coupled"
    saved_prompts = db.get_setting("vlm_prompts", {})
    config.vlm_prompts = {
        "system": (
            saved_prompts.get("system")
            if isinstance(saved_prompts, dict) and isinstance(saved_prompts.get("system"), str)
            else DEFAULT_SYSTEM_PROMPT
        ),
        "user": (
            saved_prompts.get("user")
            if isinstance(saved_prompts, dict) and isinstance(saved_prompts.get("user"), str)
            else DEFAULT_USER_PROMPT
        ),
    }
    saved_concurrency = db.get_setting("cycle_concurrency", config.concurrency)
    if (
        isinstance(saved_concurrency, int)
        and not isinstance(saved_concurrency, bool)
        and 1 <= saved_concurrency <= 24
    ):
        config.concurrency = saved_concurrency
    saved_capture_concurrency = db.get_setting(
        "cycle_capture_concurrency", config.capture_concurrency
    )
    if (
        isinstance(saved_capture_concurrency, int)
        and not isinstance(saved_capture_concurrency, bool)
        and 1 <= saved_capture_concurrency <= 24
    ):
        config.capture_concurrency = saved_capture_concurrency
    saved_inference_concurrency = db.get_setting(
        "cycle_inference_concurrency", config.inference_concurrency
    )
    if (
        isinstance(saved_inference_concurrency, int)
        and not isinstance(saved_inference_concurrency, bool)
        and 1 <= saved_inference_concurrency <= 24
    ):
        config.inference_concurrency = saved_inference_concurrency
    saved_sample_size = db.get_setting("cycle_sample_size", config.sample_size)
    if (
        isinstance(saved_sample_size, int)
        and not isinstance(saved_sample_size, bool)
        and 0 <= saved_sample_size <= 100000
    ):
        config.sample_size = saved_sample_size
    saved_interval = db.get_setting("cycle_capture_interval", config.capture_interval)
    if (
        isinstance(saved_interval, (int, float))
        and not isinstance(saved_interval, bool)
        and math.isfinite(saved_interval)
        and 5 <= saved_interval <= 300
    ):
        config.capture_interval = float(saved_interval)
    # 每次启动都重置巡检状态，避免读到上一次进程残留的“running”。
    db.set_cycle("state", "idle")
    db.set_cycle("total", 0)
    db.set_cycle("processed", 0)
    db.set_cycle("started_at", "")
    db.set_cycle("finished_at", "")
    db.set_cycle("error", "")
    latest_batch = db.get_latest_batch()
    if latest_batch and latest_batch.get("state") == "running":
        interrupted_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        db.update_batch(
            latest_batch["id"],
            state="error",
            finished_at=interrupted_at,
            error="服务重启，上一批次处理已中断",
        )
        latest_batch = db.get_batch(latest_batch["id"])
    db.set_cycle("batch_id", latest_batch["id"] if latest_batch else "")
    if config.demo_mode and not db.list_cameras():
        seed_demo(db, config)
    elif not config.demo_mode and pipeline.hik.configured:
        asyncio.create_task(pipeline.ensure_cameras())


class ReviewPayload(BaseModel):
    decision: str
    note: str = ""
    set_baseline: bool = False
    reviewer: str = "human"


class SettingsPayload(BaseModel):
    thresholds: Optional[Dict[str, Any]] = None
    module_switches: Optional[Dict[str, Any]] = None
    vlm_prompts: Optional[Dict[str, Any]] = None
    concurrency: Optional[Any] = None
    capture_interval: Optional[Any] = None
    pipeline_mode: Optional[Any] = None
    capture_concurrency: Optional[Any] = None
    inference_concurrency: Optional[Any] = None
    sample_size: Optional[Any] = None


def _camera_brief(camera: Dict[str, Any]) -> Dict[str, Any]:
    meta = camera.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return {
        "id": camera["id"],
        "device_serial": camera["device_serial"],
        "channel_no": camera["channel_no"],
        "channel_name": camera["channel_name"],
        "store_id": camera["store_id"],
        "group_no": camera["group_no"],
        "scene_template": camera["scene_template"],
        "status": camera["status"],
        "meta": meta,
        "created_at": camera["created_at"],
    }


def _occlusion_path(capture: Dict[str, Any]) -> Optional[Path]:
    modules = capture.get("module_results") or {}
    occlusion = modules.get("occlusion") if isinstance(modules, dict) else None
    if not isinstance(occlusion, dict) or not occlusion.get("overlay_available") or not capture.get("path"):
        return None
    path = Path(capture["path"]).with_suffix(".occlusion.png")
    return path if path.is_file() else None


def _capture_urls(capture: Dict[str, Any]) -> Dict[str, str]:
    capture_id = capture["id"]
    baseline_id = capture.get("baseline_id")
    return {
        "image_url": f"/api/captures/{capture_id}/image" if capture.get("path") else "",
        "thumb_url": f"/api/captures/{capture_id}/thumb" if capture.get("thumb_path") else "",
        "baseline_url": f"/api/baselines/{baseline_id}/image" if baseline_id else "",
        "occlusion_url": f"/api/captures/{capture_id}/occlusion" if _occlusion_path(capture) else "",
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "demo_mode": config.demo_mode,
        "vlm_enabled": config.vlm_enabled,
        "vlm_model": config.vlm_model,
        "artifact_root": str(config.artifact_root),
        "pipeline_mode": config.pipeline_mode,
        "module_switches": dict(config.module_switches),
    }


@app.get("/api/stats")
async def stats(batch_id: Optional[int] = None) -> Dict[str, Any]:
    base = db.stats(batch_id)
    batch = db.get_batch(batch_id) if batch_id is not None else db.get_latest_batch()
    cycle = {
        "batch_id": batch["id"] if batch else None,
        "state": db.get_cycle("state", "idle"),
        "total": db.get_cycle("total", 0),
        "processed": db.get_cycle("processed", 0),
        "started_at": db.get_cycle("started_at", ""),
        "finished_at": db.get_cycle("finished_at", ""),
    }
    base["cycle"] = cycle
    base["batch"] = batch
    return base


@app.get("/api/batches")
async def batches(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    rows = db.list_batches(limit=min(max(limit, 1), 100), offset=max(offset, 0))
    return {"rows": rows}


@app.get("/api/cameras")
async def cameras(
    status: Optional[str] = "active",
    batch_id: Optional[int] = None,
    decision: Optional[str] = None,
    query: str = "",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    rows = db.list_cameras(status)
    if not config.demo_mode:
        rows = [c for c in rows if not _camera_brief(c).get("meta", {}).get("demo")]
    if config.region_filter:
        rows = [
            c
            for c in rows
            if config.region_filter in _camera_brief(c).get("meta", {}).get("store_name", "")
        ]
    result = []
    for cam in rows:
        brief = _camera_brief(cam)
        last = db.list_captures(camera_id=cam["id"], batch_id=batch_id, limit=1)
        brief["last_capture"] = last[0] if last else None
        last_decision = last[0]["decision"] if last else "pending"
        if decision and last_decision != decision:
            continue
        needle = query.strip().lower()
        haystack = " ".join([
            str(brief.get("channel_name", "")),
            str(brief.get("device_serial", "")),
            str(brief.get("channel_no", "")),
            str(brief.get("scene_template", "")),
            str(brief.get("meta", {}).get("store_name", "")),
            str(brief.get("meta", {}).get("store_no", "")),
        ]).lower()
        if needle and needle not in haystack:
            continue
        result.append(brief)
    total = len(result)
    page_size = min(max(limit, 1), 100)
    page_offset = max(offset, 0)
    return {
        "total": total,
        "limit": page_size,
        "offset": page_offset,
        "batch_id": batch_id,
        "rows": result[page_offset:page_offset + page_size],
    }


@app.get("/api/cameras/{camera_id}/baseline")
async def camera_baseline(camera_id: int) -> Dict[str, Any]:
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="camera not found")
    baseline = db.get_latest_baseline(camera_id)
    if not baseline:
        return {"baseline": None}
    return {"baseline": {"id": baseline["id"], "kind": baseline["kind"], "url": f"/api/baselines/{baseline['id']}/image"}}


@app.get("/api/baselines/{baseline_id}/image")
async def baseline_image(baseline_id: int) -> FileResponse:
    with db._lock:
        row = db._conn.execute("SELECT * FROM baselines WHERE id=?", (baseline_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="baseline not found")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="baseline file missing")
    return FileResponse(path)


@app.get("/api/captures")
async def captures(
    decision: Optional[str] = None,
    camera_id: Optional[int] = None,
    batch_id: Optional[int] = None,
    query: str = "",
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    page_size = min(max(limit, 1), 100)
    page_offset = max(offset, 0)
    rows = db.list_captures(
        decision=decision,
        camera_id=camera_id,
        batch_id=batch_id,
        query=query,
        limit=page_size,
        offset=page_offset,
    )
    enriched = []
    for row in rows:
        cam = db.get_camera(row["camera_id"])
        row["camera"] = _camera_brief(cam) if cam else None
        row.update(_capture_urls(row))
        enriched.append(row)
    return {
        "total": db.count_captures(
            decision=decision,
            camera_id=camera_id,
            batch_id=batch_id,
            query=query,
        ),
        "limit": page_size,
        "offset": page_offset,
        "batch_id": batch_id,
        "rows": enriched,
    }


@app.get("/api/captures/{capture_id}")
async def capture_detail(capture_id: int) -> Dict[str, Any]:
    row = db.get_capture(capture_id)
    if not row:
        raise HTTPException(status_code=404, detail="capture not found")
    cam = db.get_camera(row["camera_id"])
    row["camera"] = _camera_brief(cam) if cam else None
    row.update(_capture_urls(row))
    return row


@app.get("/api/captures/{capture_id}/occlusion")
async def capture_occlusion(capture_id: int) -> FileResponse:
    row = db.get_capture(capture_id)
    path = _occlusion_path(row) if row else None
    if path is None:
        raise HTTPException(status_code=404, detail="该抓图没有可用的遮挡区域图")
    return FileResponse(path, media_type="image/png")


@app.get("/api/captures/{capture_id}/image")
async def capture_image(capture_id: int) -> FileResponse:
    row = db.get_capture(capture_id)
    if not row or not row["path"]:
        raise HTTPException(status_code=404, detail="image not found")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path)


@app.get("/api/captures/{capture_id}/thumb")
async def capture_thumb(capture_id: int) -> FileResponse:
    row = db.get_capture(capture_id)
    if not row or not row["thumb_path"]:
        raise HTTPException(status_code=404, detail="thumb not found")
    path = Path(row["thumb_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="thumb file missing")
    return FileResponse(path)


@app.post("/api/captures/{capture_id}/review")
async def review_capture(capture_id: int, payload: ReviewPayload) -> Dict[str, Any]:
    row = db.get_capture(capture_id)
    if not row:
        raise HTTPException(status_code=404, detail="capture not found")
    if payload.decision not in {"pass", "fail"}:
        raise HTTPException(status_code=400, detail="decision must be pass or fail")
    if payload.set_baseline and payload.decision != "pass":
        raise HTTPException(status_code=400, detail="只有复核合格的图片可以设为基准图")
    if payload.set_baseline and (not row["path"] or not Path(row["path"]).is_file()):
        raise HTTPException(status_code=400, detail="抓图文件不存在，无法设为基准图")
    db.add_review(
        capture_id=capture_id,
        decision=payload.decision,
        note=payload.note,
        reviewer=payload.reviewer,
        set_baseline=payload.set_baseline,
    )
    # 复核通过且要求设基准：把当前图设为该相机新基准。
    if payload.set_baseline and row["path"]:
        db.add_baseline(row["camera_id"], row["path"], kind="day", meta={"from_review": True})
    # 复核结果覆盖原判定，便于统计人工口径。
    db.update_capture_result(
        capture_id,
        payload.decision,
        1.0,
        [f"人工复核: {payload.decision}"],
        row["metrics"],
        row["module_results"],
        row.get("baseline_id"),
    )
    return {"ok": True}


@app.post("/api/cycle/start")
async def start_cycle() -> Dict[str, Any]:
    if pipeline._running:
        return {"status": "already_running"}
    asyncio.create_task(pipeline.run_cycle())
    return {"status": "started"}


@app.get("/api/cycle/status")
async def cycle_status() -> Dict[str, Any]:
    return {
        "batch_id": db.get_cycle("batch_id", None),
        "state": db.get_cycle("state", "idle"),
        "total": db.get_cycle("total", 0),
        "processed": db.get_cycle("processed", 0),
        "started_at": db.get_cycle("started_at", ""),
        "finished_at": db.get_cycle("finished_at", ""),
        "error": db.get_cycle("error", ""),
    }


@app.get("/api/cycle/progress")
async def cycle_progress() -> Dict[str, Any]:
    return pipeline.progress_snapshot()


@app.post("/api/demo/seed")
async def demo_seed() -> Dict[str, Any]:
    return seed_demo(db, config)


@app.get("/api/settings")
async def settings() -> Dict[str, Any]:
    return {
        "demo_mode": config.demo_mode,
        "concurrency": config.concurrency,
        "capture_concurrency": config.capture_concurrency,
        "inference_concurrency": config.inference_concurrency,
        "sample_size": config.sample_size,
        "capture_interval": config.capture_interval,
        "vlm_enabled": config.vlm_enabled,
        "vlm_model": config.vlm_model,
        "artifact_root": str(config.artifact_root),
        "pipeline_mode": config.pipeline_mode,
        "module_switches": dict(config.module_switches),
        "vlm_prompts": dict(config.vlm_prompts),
        "thresholds": config.thresholds,
    }


@app.put("/api/settings")
async def update_settings(payload: SettingsPayload) -> Dict[str, Any]:
    updated_thresholds = config.thresholds
    updated_modules = dict(config.module_switches)
    updated_prompts = dict(config.vlm_prompts)
    updated_concurrency = config.concurrency
    updated_capture_interval = config.capture_interval
    updated_pipeline_mode = config.pipeline_mode
    updated_capture_concurrency = config.capture_concurrency
    updated_inference_concurrency = config.inference_concurrency
    updated_sample_size = config.sample_size
    if payload.pipeline_mode is not None:
        try:
            updated_pipeline_mode = validate_pipeline_mode(payload.pipeline_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    for field_name, value in (
        ("capture_concurrency", payload.capture_concurrency),
        ("inference_concurrency", payload.inference_concurrency),
    ):
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 24
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} 必须是 1–24 的整数",
            )
        if field_name == "capture_concurrency":
            updated_capture_concurrency = value
        else:
            updated_inference_concurrency = value
    if payload.concurrency is not None:
        if (
            isinstance(payload.concurrency, bool)
            or not isinstance(payload.concurrency, int)
            or not 1 <= payload.concurrency <= 24
        ):
            raise HTTPException(status_code=400, detail="同时推理图片数必须是 1–24 的整数")
        updated_concurrency = payload.concurrency
    if payload.sample_size is not None:
        if (
            isinstance(payload.sample_size, bool)
            or not isinstance(payload.sample_size, int)
            or not 0 <= payload.sample_size <= 100000
        ):
            raise HTTPException(
                status_code=400,
                detail="每轮抽样数量必须是 0–100000 的整数，0 表示全部",
            )
        updated_sample_size = payload.sample_size
    if payload.capture_interval is not None:
        if (
            isinstance(payload.capture_interval, bool)
            or not isinstance(payload.capture_interval, (int, float))
            or not math.isfinite(payload.capture_interval)
            or not 5 <= payload.capture_interval <= 300
        ):
            raise HTTPException(status_code=400, detail="抓图间隔必须是 5–300 秒之间的数字")
        updated_capture_interval = float(payload.capture_interval)
    if payload.module_switches is not None:
        if payload.pipeline_mode is None:
            raise HTTPException(
                status_code=400,
                detail="模块开关已由检测链路替代，请提交 pipeline_mode",
            )
        try:
            updated_modules = validate_module_switches(
                config.module_switches, payload.module_switches
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.vlm_prompts is not None:
        if not isinstance(payload.vlm_prompts, dict):
            raise HTTPException(status_code=400, detail="Prompt 配置必须是对象")
        unknown = set(payload.vlm_prompts) - {"system", "user"}
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"未知 Prompt 字段：{', '.join(sorted(unknown))}",
            )
        for key in ("system", "user"):
            if key not in payload.vlm_prompts:
                continue
            value = payload.vlm_prompts[key]
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(status_code=400, detail=f"{key} Prompt 不能为空")
            if len(value) > 30000:
                raise HTTPException(status_code=400, detail=f"{key} Prompt 不能超过 30000 字符")
            updated_prompts[key] = value.strip()
    if payload.thresholds:
        try:
            candidate = validate_thresholds(config.thresholds, payload.thresholds)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        updated_thresholds = candidate
    # 阈值和开关全部验证通过后再原子地写入内存与数据库。
    if payload.thresholds:
        db.set_setting("thresholds", updated_thresholds)
        config.thresholds = updated_thresholds
    if payload.module_switches is not None:
        db.set_setting("legacy_module_switches", updated_modules)
    if payload.vlm_prompts is not None:
        db.set_setting("vlm_prompts", updated_prompts)
        config.vlm_prompts = updated_prompts
    if payload.concurrency is not None:
        db.set_setting("cycle_concurrency", updated_concurrency)
        config.concurrency = updated_concurrency
    if payload.capture_interval is not None:
        db.set_setting("cycle_capture_interval", updated_capture_interval)
        config.capture_interval = updated_capture_interval
    if payload.pipeline_mode is not None:
        db.set_setting("pipeline_mode", updated_pipeline_mode)
        config.pipeline_mode = updated_pipeline_mode
    if payload.capture_concurrency is not None:
        db.set_setting("cycle_capture_concurrency", updated_capture_concurrency)
        config.capture_concurrency = updated_capture_concurrency
    if payload.inference_concurrency is not None:
        db.set_setting("cycle_inference_concurrency", updated_inference_concurrency)
        config.inference_concurrency = updated_inference_concurrency
    if payload.sample_size is not None:
        db.set_setting("cycle_sample_size", updated_sample_size)
        config.sample_size = updated_sample_size
    return {
        "ok": True,
        "thresholds": config.thresholds,
        "module_switches": dict(config.module_switches),
        "pipeline_mode": config.pipeline_mode,
        "vlm_prompts": dict(config.vlm_prompts),
        "concurrency": config.concurrency,
        "capture_concurrency": config.capture_concurrency,
        "inference_concurrency": config.inference_concurrency,
        "sample_size": config.sample_size,
        "capture_interval": config.capture_interval,
    }
