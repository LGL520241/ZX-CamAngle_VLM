"""Repeatable prompt regression against reviewed production captures."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cam_angle.config import Config, restore_module_switches, restore_thresholds
from cam_angle.database import Database
from cam_angle.decision import fuse_semantic_decision
from cam_angle.vlm import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, QwenSemanticClient


CORE_CASES = [
    (16456, "invalid", "near-lens foreground"),
    (16408, "invalid", "global haze"),
    (16428, "invalid", "wall-only"),
    (16420, "invalid", "foreground stack"),
    (15972, "valid", "outdoor seating"),
    (16350, "valid", "empty kitchen"),
    (16066, "valid", "crossing cable"),
    (15780, "valid", "dining"),
    (14876, "valid", "empty prep area"),
    (14852, "valid", "empty cashier"),
]

FULL_CASES = CORE_CASES + [
    (15791, "valid", "kitchen"),
    (15059, "valid", "kitchen"),
    (15885, "valid", "kitchen"),
    (16074, "valid", "kitchen-cashier"),
    (15379, "valid", "kitchen"),
    (15141, "invalid", "low-texture blur"),
    (15549, "invalid", "low-texture blur"),
    (15674, "invalid", "low-texture blur"),
    (15761, "invalid", "low-texture blur"),
]


def cv_result() -> Dict[str, Any]:
    return {
        "metrics": {},
        "module_results": {
            "quality": {"status": "pass", "confidence": 0.95, "reason": "prompt replay"},
            "occlusion": {
                "status": "skipped",
                "confidence": 0.0,
                "reason": "disabled during prompt replay",
                "metrics": {"occlusion_ratio": 0.0},
            },
            "blur": {"status": "pass", "confidence": 0.9, "reason": "prompt replay"},
            "rotation": {"status": "pass", "confidence": 0.9, "reason": "prompt replay"},
        },
        "baseline_id": None,
    }


async def run_case(
    client: QwenSemanticClient,
    config: Config,
    conn: sqlite3.Connection,
    semaphore: asyncio.Semaphore,
    case: tuple[int, str, str],
    prompts: Dict[str, str],
) -> Dict[str, Any]:
    capture_id, expected, label = case
    row = conn.execute(
        "SELECT path, camera_id FROM captures WHERE id=?", (capture_id,)
    ).fetchone()
    if not row or not row["path"]:
        return {
            "capture_id": capture_id,
            "expected": expected,
            "label": label,
            "actual": "missing",
        }
    camera = conn.execute(
        "SELECT * FROM cameras WHERE id=?", (row["camera_id"],)
    ).fetchone()
    camera_data = dict(camera) if camera else {"channel_name": "prompt replay", "meta": "{}"}
    async with semaphore:
        result = await client.analyze(Path(row["path"]), camera_data, prompts)
    semantic = result.get("data") or {}
    decision = (
        fuse_semantic_decision(cv_result(), result, config)
        if semantic
        else {"decision": "error", "reasons": [result.get("reason")]}
    )
    return {
        "capture_id": capture_id,
        "expected": expected,
        "label": label,
        "actual": decision["decision"],
        "reasons": decision["reasons"],
        "semantic": {
            key: semantic.get(key)
            for key in (
                "camera_type",
                "target_area_visible",
                "view_valid",
                "view_issue",
                "subject_coverage",
                "occlusion",
                "occlusion_area_ratio",
                "occlusion_scope",
                "occlusion_affects_subject",
                "blur",
                "blur_scope",
                "monitoring_subject_affected",
                "obstruction_verified",
            )
        },
        "focus": (result.get("metrics") or {}).get("vlm_foreground_focus_suspicion"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("core", "full"), default="core")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    db = Database(config.db_path)
    config.thresholds = restore_thresholds(
        config.thresholds, db.get_setting("thresholds", {})
    )
    config.module_switches = restore_module_switches(
        db.get_setting("module_switches", {})
    )
    config.vlm_enabled = True
    prompts = {"system": DEFAULT_SYSTEM_PROMPT, "user": DEFAULT_USER_PROMPT}
    cases = FULL_CASES if args.set == "full" else CORE_CASES
    conn = sqlite3.connect(str(config.db_path))
    conn.row_factory = sqlite3.Row
    client = QwenSemanticClient(config)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(
        *(run_case(client, config, conn, semaphore, case, prompts) for case in cases)
    )
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    passed = sum(
        (item["expected"] == "valid") == (item["actual"] == "pass")
        for item in results
    )
    print(f"REGRESSION {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
