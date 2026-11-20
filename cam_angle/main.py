from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_config
from .database import Database
from .demo import seed_demo
from .pipeline import Pipeline


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
    # 每次启动都重置巡检状态，避免读到上一次进程残留的“running”。
    db.set_cycle("state", "idle")
    db.set_cycle("total", 0)
    db.set_cycle("processed", 0)
    db.set_cycle("started_at", "")
    db.set_cycle("finished_at", "")
    db.set_cycle("error", "")
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
    thresholds: Optional[Dict[str, float]] = None


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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "demo_mode": config.demo_mode}


@app.get("/api/stats")
async def stats() -> Dict[str, Any]:
    base = db.stats()
    cycle = {
        "state": db.get_cycle("state", "idle"),
        "total": db.get_cycle("total", 0),
        "processed": db.get_cycle("processed", 0),
        "started_at": db.get_cycle("started_at", ""),
        "finished_at": db.get_cycle("finished_at", ""),
    }
    base["cycle"] = cycle
    return base


@app.get("/api/cameras")
async def cameras(status: Optional[str] = "active", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = db.list_cameras(status)
    if not config.demo_mode:
        rows = [c for c in rows if not _camera_brief(c).get("meta", {}).get("demo")]
    if config.region_filter:
        rows = [
            c
            for c in rows
            if config.region_filter in _camera_brief(c).get("meta", {}).get("store_name", "")
        ]
    if limit is not None:
        rows = rows[:limit]
    result = []
    for cam in rows:
        brief = _camera_brief(cam)
        last = db.list_captures(camera_id=cam["id"], limit=1)
        brief["last_capture"] = last[0] if last else None
        result.append(brief)
    return result


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
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    rows = db.list_captures(decision=decision, camera_id=camera_id, limit=min(limit, 500), offset=offset)
    enriched = []
    for row in rows:
        cam = db.get_camera(row["camera_id"])
        row["camera"] = _camera_brief(cam) if cam else None
        row["image_url"] = f"/api/captures/{row['id']}/image" if row["path"] else ""
        row["thumb_url"] = f"/api/captures/{row['id']}/thumb" if row["thumb_path"] else ""
        enriched.append(row)
    return {"total": db.count_captures(decision), "rows": enriched}


@app.get("/api/captures/{capture_id}")
async def capture_detail(capture_id: int) -> Dict[str, Any]:
    row = db.get_capture(capture_id)
    if not row:
        raise HTTPException(status_code=404, detail="capture not found")
    cam = db.get_camera(row["camera_id"])
    row["camera"] = _camera_brief(cam) if cam else None
    row["image_url"] = f"/api/captures/{row['id']}/image" if row["path"] else ""
    row["thumb_url"] = f"/api/captures/{row['id']}/thumb" if row["thumb_path"] else ""
    baseline = db.get_latest_baseline(row["camera_id"])
    row["baseline_url"] = f"/api/baselines/{baseline['id']}/image" if baseline else ""
    return row


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
        "capture_interval": config.capture_interval,
        "thresholds": config.thresholds,
    }


@app.put("/api/settings")
async def update_settings(payload: SettingsPayload) -> Dict[str, Any]:
    if payload.thresholds:
        config.thresholds.update(payload.thresholds)
        db.set_setting("thresholds", config.thresholds)
    return {"ok": True, "thresholds": config.thresholds}
