from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional


PERMISSIONS = (
    "users.manage",
    "settings.write",
    "cycle.control",
    "review.write",
    "maintenance.run",
)

ROLE_PERMISSIONS = {
    "admin": list(PERMISSIONS),
    "operator": ["cycle.control", "review.write"],
    "reviewer": ["review.write"],
    "viewer": [],
}

ROLE_LABELS = {
    "admin": "系统管理员",
    "operator": "巡检操作员",
    "reviewer": "审核员",
    "viewer": "只读查看员",
}

PBKDF2_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("密码至少需要 8 个字符")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}$"
        f"{salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(rounds),
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def normalize_permissions(role: str, requested: Optional[Iterable[str]] = None) -> List[str]:
    if role not in ROLE_PERMISSIONS:
        raise ValueError("未知用户角色")
    if role == "admin":
        return list(PERMISSIONS)
    defaults = ROLE_PERMISSIONS[role]
    values = defaults if requested is None else requested
    permissions = sorted({str(item) for item in values if str(item) in PERMISSIONS})
    return permissions


def issue_session_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, hash_session_token(token)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiry(hours: int) -> str:
    return (
        datetime.now(timezone.utc).astimezone() + timedelta(hours=max(1, hours))
    ).isoformat(timespec="seconds")


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "role": user["role"],
        "role_label": ROLE_LABELS.get(user["role"], user["role"]),
        "permissions": list(user.get("permissions") or []),
        "status": user["status"],
        "created_at": user.get("created_at", ""),
        "last_login_at": user.get("last_login_at", ""),
    }


def has_permission(user: Optional[Dict[str, Any]], permission: str) -> bool:
    if not user:
        return False
    if user.get("role") == "admin":
        return True
    return permission in (user.get("permissions") or [])
