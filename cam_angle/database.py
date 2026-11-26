from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Database:
    """基于 SQLite 的轻量存储，后续可替换为 PostgreSQL。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_serial TEXT NOT NULL,
                    channel_no INTEGER NOT NULL,
                    channel_name TEXT DEFAULT '',
                    group_no TEXT DEFAULT '',
                    store_id TEXT DEFAULT '',
                    scene_template TEXT DEFAULT 'default',
                    status TEXT DEFAULT 'active',
                    meta TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(device_serial, channel_no)
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT DEFAULT 'day',
                    meta TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(camera_id) REFERENCES cameras(id)
                );

                CREATE TABLE IF NOT EXISTS inspection_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT DEFAULT '',
                    state TEXT DEFAULT 'running',
                    total INTEGER DEFAULT 0,
                    processed INTEGER DEFAULT 0,
                    error TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS captures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER,
                    camera_id INTEGER NOT NULL,
                    path TEXT DEFAULT '',
                    thumb_path TEXT DEFAULT '',
                    source TEXT DEFAULT 'capture',
                    quality INTEGER DEFAULT 0,
                    capture_time TEXT NOT NULL,
                    analyzed_at TEXT DEFAULT '',
                    decision TEXT DEFAULT 'pending',
                    confidence REAL DEFAULT 0.0,
                    reasons TEXT DEFAULT '[]',
                    metrics TEXT DEFAULT '{}',
                    module_results TEXT DEFAULT '{}',
                    baseline_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(camera_id) REFERENCES cameras(id),
                    FOREIGN KEY(batch_id) REFERENCES inspection_batches(id)
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_id INTEGER NOT NULL UNIQUE,
                    reviewer TEXT DEFAULT 'human',
                    decision TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    set_baseline INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(capture_id) REFERENCES captures(id)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cycle_status (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT DEFAULT '',
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    permissions TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    username TEXT DEFAULT '',
                    action TEXT NOT NULL,
                    target TEXT DEFAULT '',
                    detail TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            capture_columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(captures)").fetchall()
            }
            if "batch_id" not in capture_columns:
                self._conn.execute("ALTER TABLE captures ADD COLUMN batch_id INTEGER")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_captures_batch_id ON captures(batch_id, id DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_captures_camera_batch ON captures(camera_id, batch_id, id DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id, expires_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cameras_store_no "
                "ON cameras(json_extract(meta, '$.store_no'))"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cameras_store_name "
                "ON cameras(json_extract(meta, '$.store_name'))"
            )

    # ---------- 通用 ----------
    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _loads(self, value: Optional[str], default: Any) -> Any:
        if value is None:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, self._json(value)),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return self._loads(row["value"], default)

    def set_cycle(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO cycle_status(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, self._json(value)),
            )

    def get_cycle(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cycle_status WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return self._loads(row["value"], default)

    # ---------- 抓帧批次 ----------
    def create_batch(self, started_at: str) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO inspection_batches(started_at, state, created_at)
                VALUES(?, 'running', ?)
                """,
                (started_at, now),
            )
            return int(cur.lastrowid)

    def update_batch(
        self,
        batch_id: int,
        *,
        state: Optional[str] = None,
        total: Optional[int] = None,
        processed: Optional[int] = None,
        finished_at: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        updates = []
        params: list = []
        for column, value in (
            ("state", state),
            ("total", total),
            ("processed", processed),
            ("finished_at", finished_at),
            ("error", error),
        ):
            if value is not None:
                updates.append(f"{column}=?")
                params.append(value)
        if not updates:
            return
        params.append(batch_id)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE inspection_batches SET {', '.join(updates)} WHERE id=?",
                params,
            )

    def get_batch(self, batch_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM inspection_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_batch(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM inspection_batches ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def list_batches(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT b.*,
                       COUNT(c.id) AS capture_count,
                       COALESCE(SUM(CASE WHEN c.decision='pass' THEN 1 ELSE 0 END), 0) AS pass_count,
                       COALESCE(SUM(CASE WHEN c.decision='fail' THEN 1 ELSE 0 END), 0) AS fail_count,
                       COALESCE(SUM(CASE WHEN c.decision='review' THEN 1 ELSE 0 END), 0) AS review_count,
                       COALESCE(SUM(CASE WHEN c.decision='no_signal' THEN 1 ELSE 0 END), 0) AS no_signal_count
                FROM inspection_batches b
                LEFT JOIN captures c ON c.batch_id=b.id
                GROUP BY b.id
                ORDER BY b.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------- 相机 ----------
    def upsert_camera(
        self,
        device_serial: str,
        channel_no: int,
        channel_name: str = "",
        group_no: str = "",
        store_id: str = "",
        scene_template: str = "default",
        status: str = "active",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO cameras(
                    device_serial, channel_no, channel_name, group_no, store_id,
                    scene_template, status, meta, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(device_serial, channel_no) DO UPDATE SET
                    channel_name=excluded.channel_name,
                    group_no=excluded.group_no,
                    store_id=excluded.store_id,
                    scene_template=excluded.scene_template,
                    status=excluded.status,
                    meta=excluded.meta
                """,
                (
                    device_serial,
                    channel_no,
                    channel_name,
                    group_no,
                    store_id,
                    scene_template,
                    status,
                    self._json(meta or {}),
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT id FROM cameras WHERE device_serial=? AND channel_no=?",
                (device_serial, channel_no),
            ).fetchone()
            return int(row["id"])

    def get_camera(self, camera_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
        return dict(row) if row else None

    def list_cameras(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM cameras"
        params: tuple = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_store_context(
        self, camera_id: int, batch_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        camera = self.get_camera(camera_id)
        if not camera:
            return []
        meta = self._loads(camera.get("meta"), {})
        store_no = str(meta.get("store_no") or "")
        store_name = str(meta.get("store_name") or "")
        if not store_no and not store_name:
            return []
        clauses = ["cam.status='active'"]
        params: list = []
        if store_no:
            clauses.append("COALESCE(json_extract(cam.meta, '$.store_no'), '')=?")
            params.append(store_no)
        else:
            clauses.append("COALESCE(json_extract(cam.meta, '$.store_name'), '')=?")
            params.append(store_name)
        capture_batch_clause = ""
        capture_params: list = []
        if batch_id is not None:
            capture_batch_clause = " AND c.batch_id=?"
            capture_params.append(batch_id)
        sql = f"""
            SELECT cam.id AS camera_id, cam.channel_name, cam.device_serial,
                   cam.channel_no, cam.meta,
                   c.id AS capture_id, c.decision, c.confidence, c.capture_time,
                   c.thumb_path, c.path
            FROM cameras cam
            LEFT JOIN captures c ON c.id = (
                SELECT c2.id FROM captures c2
                WHERE c2.camera_id=cam.id{capture_batch_clause}
                ORDER BY c2.id DESC LIMIT 1
            )
            WHERE {' AND '.join(clauses)}
            ORDER BY cam.channel_no, cam.id
        """
        with self._lock:
            rows = self._conn.execute(sql, capture_params + params).fetchall()
        return [dict(row) for row in rows]

    # ---------- 基准图 ----------
    def add_baseline(
        self, camera_id: int, path: str, kind: str = "day", meta: Optional[Dict[str, Any]] = None
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO baselines(camera_id, path, kind, meta, created_at) VALUES(?,?,?,?,?)",
                (camera_id, path, kind, self._json(meta or {}), now),
            )
            return int(cur.lastrowid)

    def get_latest_baseline(self, camera_id: int, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM baselines WHERE camera_id=?"
        params: tuple = (camera_id,)
        if kind:
            sql += " AND kind=?"
            params = (camera_id, kind)
        sql += " ORDER BY id DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # ---------- 抓图记录 ----------
    def add_capture(
        self,
        camera_id: int,
        path: str,
        thumb_path: str,
        source: str,
        quality: int,
        capture_time: str,
        decision: str = "pending",
        confidence: float = 0.0,
        reasons: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        module_results: Optional[Dict[str, Any]] = None,
        baseline_id: Optional[int] = None,
        batch_id: Optional[int] = None,
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO captures(
                    batch_id, camera_id, path, thumb_path, source, quality, capture_time,
                    analyzed_at, decision, confidence, reasons, metrics, module_results,
                    baseline_id, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    batch_id,
                    camera_id,
                    path,
                    thumb_path,
                    source,
                    quality,
                    capture_time,
                    now,
                    decision,
                    confidence,
                    self._json(reasons or []),
                    self._json(metrics or {}),
                    self._json(module_results or {}),
                    baseline_id,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def update_capture_result(
        self,
        capture_id: int,
        decision: str,
        confidence: float,
        reasons: List[str],
        metrics: Dict[str, Any],
        module_results: Dict[str, Any],
        baseline_id: Optional[int] = None,
    ) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE captures SET analyzed_at=?, decision=?, confidence=?,
                    reasons=?, metrics=?, module_results=?, baseline_id=COALESCE(?, baseline_id)
                WHERE id=?
                """,
                (
                    now,
                    decision,
                    confidence,
                    self._json(reasons),
                    self._json(metrics),
                    self._json(module_results),
                    baseline_id,
                    capture_id,
                ),
            )

    def get_capture(self, capture_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM captures WHERE id=?", (capture_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["reasons"] = self._loads(data.get("reasons"), [])
        data["metrics"] = self._loads(data.get("metrics"), {})
        data["module_results"] = self._loads(data.get("module_results"), {})
        return data

    def list_captures(
        self,
        decision: Optional[str] = None,
        camera_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        query: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where, params = self._capture_filters(
            decision=decision,
            camera_id=camera_id,
            batch_id=batch_id,
            query=query,
        )
        sql = (
            "SELECT c.* FROM captures c "
            "LEFT JOIN cameras cam ON cam.id=c.camera_id "
            f"WHERE {where} ORDER BY c.id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["reasons"] = self._loads(data.get("reasons"), [])
            data["metrics"] = self._loads(data.get("metrics"), {})
            data["module_results"] = self._loads(data.get("module_results"), {})
            result.append(data)
        return result

    def list_capture_summaries(
        self,
        decision: Optional[str] = None,
        camera_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        query: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """列表场景只取卡片需要的字段，避免传输大型模块结果。"""
        where, params = self._capture_filters(
            decision=decision,
            camera_id=camera_id,
            batch_id=batch_id,
            query=query,
        )
        sql = (
            "SELECT c.id, c.batch_id, c.camera_id, c.path, c.thumb_path, "
            "c.source, c.quality, c.capture_time, c.decision, c.confidence, "
            "c.reasons, c.baseline_id "
            "FROM captures c "
            "LEFT JOIN cameras cam ON cam.id=c.camera_id "
            f"WHERE {where} ORDER BY c.id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["reasons"] = self._loads(data.get("reasons"), [])
            result.append(data)
        return result

    def _capture_filters(
        self,
        *,
        decision: Optional[str] = None,
        camera_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        query: str = "",
    ) -> tuple[str, list]:
        clauses = ["1=1"]
        params: list = []
        if decision:
            clauses.append("c.decision=?")
            params.append(decision)
        if camera_id:
            clauses.append("c.camera_id=?")
            params.append(camera_id)
        if batch_id is not None:
            clauses.append("c.batch_id=?")
            params.append(batch_id)
        needle = query.strip()
        if needle:
            like = f"%{needle}%"
            clauses.append(
                "(cam.channel_name LIKE ? OR cam.device_serial LIKE ? "
                "OR COALESCE(json_extract(cam.meta, '$.store_name'), '') LIKE ? "
                "OR c.reasons LIKE ?)"
            )
            params.extend([like, like, like, like])
        return " AND ".join(clauses), params

    def count_captures(
        self,
        decision: Optional[str] = None,
        camera_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        query: str = "",
    ) -> int:
        where, params = self._capture_filters(
            decision=decision,
            camera_id=camera_id,
            batch_id=batch_id,
            query=query,
        )
        sql = (
            "SELECT COUNT(*) AS c FROM captures c "
            "LEFT JOIN cameras cam ON cam.id=c.camera_id "
            f"WHERE {where}"
        )
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["c"])

    # ---------- 复核 ----------
    def add_review(
        self,
        capture_id: int,
        decision: str,
        note: str = "",
        reviewer: str = "human",
        set_baseline: bool = False,
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO reviews(capture_id, reviewer, decision, note, set_baseline, created_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(capture_id) DO UPDATE SET
                    reviewer=excluded.reviewer,
                    decision=excluded.decision,
                    note=excluded.note,
                    set_baseline=excluded.set_baseline,
                    created_at=excluded.created_at
                """,
                (capture_id, reviewer, decision, note, 1 if set_baseline else 0, now),
            )
            return int(cur.lastrowid)

    def stats(self, batch_id: Optional[int] = None) -> Dict[str, Any]:
        where = " WHERE batch_id=?" if batch_id is not None else ""
        params: tuple = (batch_id,) if batch_id is not None else ()
        with self._lock:
            total_cameras = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS c FROM cameras WHERE status='active'"
                ).fetchone()["c"]
            )
            total_captures = int(
                self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM captures{where}",
                    params,
                ).fetchone()["c"]
            )
            by_decision_rows = self._conn.execute(
                f"SELECT decision, COUNT(*) AS c FROM captures{where} GROUP BY decision",
                params,
            ).fetchall()
            review_count = int(
                self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM captures{where + (' AND ' if where else ' WHERE ')}decision='review'",
                    params,
                ).fetchone()["c"]
            )
        by_decision = {r["decision"]: int(r["c"]) for r in by_decision_rows}
        return {
            "total_cameras": total_cameras,
            "total_captures": total_captures,
            "review_count": review_count,
            "by_decision": by_decision,
        }

    # ---------- 用户、会话与审计 ----------
    def count_users(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        return int(row["c"])

    def create_user(
        self,
        username: str,
        password_hash: str,
        display_name: str,
        role: str,
        permissions: List[str],
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO users(
                    username, display_name, password_hash, role, permissions,
                    status, created_at, updated_at
                ) VALUES(?,?,?,?,?,'active',?,?)
                """,
                (
                    username,
                    display_name,
                    password_hash,
                    role,
                    self._json(permissions),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def _user_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["permissions"] = self._loads(data.get("permissions"), [])
        return data

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE",
                (username,),
            ).fetchone()
        return self._user_row(row) if row else None

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._user_row(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [self._user_row(row) for row in rows]

    def update_user(
        self,
        user_id: int,
        *,
        display_name: Optional[str] = None,
        role: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        status: Optional[str] = None,
        password_hash: Optional[str] = None,
    ) -> None:
        updates = []
        params: list = []
        for column, value in (
            ("display_name", display_name),
            ("role", role),
            ("permissions", self._json(permissions) if permissions is not None else None),
            ("status", status),
            ("password_hash", password_hash),
        ):
            if value is not None:
                updates.append(f"{column}=?")
                params.append(value)
        if not updates:
            return
        updates.append("updated_at=?")
        params.extend([_now(), user_id])
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id=?",
                params,
            )

    def touch_user_login(self, user_id: int) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET last_login_at=?, updated_at=? WHERE id=?",
                (now, now, user_id),
            )

    def create_session(self, token_hash: str, user_id: int, expires_at: str) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO user_sessions(token_hash, user_id, created_at, expires_at, last_seen_at)
                VALUES(?,?,?,?,?)
                """,
                (token_hash, user_id, now, expires_at, now),
            )

    def get_session_user(self, token_hash: str) -> Optional[Dict[str, Any]]:
        now = _now()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT u.* FROM user_sessions s
                JOIN users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.expires_at>? AND u.status='active'
                """,
                (token_hash, now),
            ).fetchone()
        return self._user_row(row) if row else None

    def touch_session(self, token_hash: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE user_sessions SET last_seen_at=? WHERE token_hash=?",
                (_now(), token_hash),
            )

    def revoke_session(self, token_hash: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM user_sessions WHERE token_hash=?", (token_hash,))

    def revoke_user_sessions(self, user_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM user_sessions WHERE user_id=?", (user_id,))

    def cleanup_sessions(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM user_sessions WHERE expires_at<=?", (_now(),))

    def add_audit(
        self,
        user_id: Optional[int],
        username: str,
        action: str,
        target: str = "",
        detail: Any = "",
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO audit_logs(user_id, username, action, target, detail, created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    user_id,
                    username,
                    action,
                    target,
                    self._json(detail) if not isinstance(detail, str) else detail,
                    _now(),
                ),
            )
