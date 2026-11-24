from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now().astimezone()


class ArtifactStore:
    """按抓帧日期和批次时间保存源图、中间结果和最终判定。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def begin_cycle(self, started_at: str, batch_id: int) -> Path:
        moment = _parse_time(started_at)
        day_dir = self.root / f"{moment:%Y}" / f"{moment:%Y%m}" / f"{moment:%Y%m%d}"
        base_name = f"{moment:%Y%m%d_%H%M%S}"
        cycle_dir = day_dir / base_name
        cycle_dir.mkdir(parents=True, exist_ok=True)
        # 同一秒再次启动时保留“抓帧时间”语义，并增加批次号避免覆盖。
        if (cycle_dir / "cycle.json").exists():
            cycle_dir = day_dir / f"{base_name}_batch{batch_id}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
        return cycle_dir

    @staticmethod
    def _safe(value: Any, fallback: str = "unknown") -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
        return text.strip("-") or fallback

    def capture_paths(
        self,
        cycle_dir: Path,
        capture_time: str,
        device_serial: str,
        channel_no: int,
    ) -> Dict[str, Path]:
        moment = _parse_time(capture_time)
        stamp = f"{moment:%Y%m%d_%H%M%S_%f}"[:-3]
        identity = f"{self._safe(device_serial)}_{self._safe(channel_no)}"
        stem = f"{stamp}__{identity}"
        return {
            "stem": cycle_dir / stem,
            "source": cycle_dir / f"{stem}.jpg",
            "thumb": cycle_dir / f"{stem}.thumb.jpg",
            "cv": cycle_dir / f"{stem}.cv.json",
            "vlm_request": cycle_dir / f"{stem}.vlm-request.json",
            "vlm_input": cycle_dir / f"{stem}.vlm-input.jpg",
            "vlm_response": cycle_dir / f"{stem}.vlm-response.json",
            "quality_guard": cycle_dir / f"{stem}.quality-guard.json",
            "analysis": cycle_dir / f"{stem}.analysis.json",
            "error": cycle_dir / f"{stem}.error.json",
        }

    def write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    def append_event(self, cycle_dir: Path, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with (cycle_dir / "run.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_cycle(self, cycle_dir: Path, value: Dict[str, Any]) -> None:
        self.write_json(cycle_dir / "cycle.json", value)
