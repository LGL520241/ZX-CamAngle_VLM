from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List


DECISIONS = ("pass", "fail", "review", "no_signal", "pending")


def _loads(value: Any, default: Any) -> Any:
    import json

    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def build_batch_report(db: Any, batch_id: int) -> Dict[str, Any]:
    batch = db.get_batch(batch_id)
    if not batch:
        raise ValueError("batch not found")
    with db._lock:
        rows = db._conn.execute(
            """
            SELECT c.*, cam.channel_name, cam.device_serial, cam.channel_no, cam.meta AS camera_meta
            FROM captures c
            LEFT JOIN cameras cam ON cam.id=c.camera_id
            WHERE c.batch_id=?
            ORDER BY c.id
            """,
            (batch_id,),
        ).fetchall()
        store_rows = db._conn.execute(
            """
            SELECT COALESCE(json_extract(cam.meta, '$.store_name'), '未标注门店') AS store_name,
                   COALESCE(json_extract(cam.meta, '$.store_no'), '') AS store_no,
                   COUNT(c.id) AS total,
                   COALESCE(SUM(CASE WHEN c.decision='pass' THEN 1 ELSE 0 END), 0) AS pass_count,
                   COALESCE(SUM(CASE WHEN c.decision='fail' THEN 1 ELSE 0 END), 0) AS fail_count,
                   COALESCE(SUM(CASE WHEN c.decision='review' THEN 1 ELSE 0 END), 0) AS review_count,
                   COALESCE(SUM(CASE WHEN c.decision='no_signal' THEN 1 ELSE 0 END), 0) AS no_signal_count
            FROM captures c
            LEFT JOIN cameras cam ON cam.id=c.camera_id
            WHERE c.batch_id=?
            GROUP BY store_name, store_no
            ORDER BY fail_count DESC, review_count DESC, total DESC
            """,
            (batch_id,),
        ).fetchall()
        reviewed = db._conn.execute(
            """
            SELECT COUNT(*) AS c FROM reviews r
            JOIN captures c ON c.id=r.capture_id
            WHERE c.batch_id=?
            """,
            (batch_id,),
        ).fetchone()["c"]

    decisions = Counter(str(row["decision"] or "pending") for row in rows)
    reasons = Counter()
    module_failures = Counter()
    timeline = Counter()
    confidence_values = []
    for row in rows:
        if row["confidence"] is not None:
            confidence_values.append(float(row["confidence"]))
        capture_time = str(row["capture_time"] or "")
        if len(capture_time) >= 16:
            timeline[capture_time[11:16]] += 1
        for reason in _loads(row["reasons"], []):
            text = str(reason).strip()
            if text:
                reasons[text] += 1
        modules = _loads(row["module_results"], {})
        if isinstance(modules, dict):
            for name, result in modules.items():
                if isinstance(result, dict) and result.get("status") == "fail":
                    module_failures[str(name)] += 1

    by_decision = {key: int(decisions.get(key, 0)) for key in DECISIONS}
    total = len(rows)
    stores: List[Dict[str, Any]] = []
    for row in store_rows:
        item = dict(row)
        item["pass_rate"] = round(item["pass_count"] * 100 / item["total"], 1) if item["total"] else 0.0
        stores.append(item)

    return {
        "batch": batch,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": {
            "total": total,
            "reviewed": int(reviewed or 0),
            "pass_rate": round(by_decision["pass"] * 100 / total, 1) if total else 0.0,
            "average_confidence": round(sum(confidence_values) / len(confidence_values), 3)
            if confidence_values
            else 0.0,
        },
        "by_decision": by_decision,
        "stores": stores,
        "top_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(12)
        ],
        "module_failures": dict(module_failures.most_common()),
        "timeline": [{"minute": minute, "count": count} for minute, count in sorted(timeline.items())],
    }
