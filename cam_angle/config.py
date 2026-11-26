from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODULE_SWITCH_KEYS = ("occlusion", "blur", "qwen_semantic")
MODULE_SWITCH_DEFAULTS = {
    "occlusion": False,
    "blur": False,
    "qwen_semantic": True,
}
PIPELINE_MODES = ("coupled", "decoupled")
PIPELINE_MODE_DEFAULTS = "coupled"


@dataclass
class Config:
    demo_mode: bool = field(default_factory=lambda: _env_bool("DEMO_MODE", True))
    hik_client_id: str = field(default_factory=lambda: os.getenv("HIK_CLIENT_ID", ""))
    hik_client_secret: str = field(default_factory=lambda: os.getenv("HIK_CLIENT_SECRET", ""))
    hik_base_url: str = field(default_factory=lambda: os.getenv("HIK_BASE_URL", "https://api2.hik-cloud.com"))
    hik_group_nos: str = field(default_factory=lambda: os.getenv("HIK_GROUP_NOS", "1"))
    region_filter: str = field(default_factory=lambda: os.getenv("REGION_FILTER", ""))
    sample_size: int = field(default_factory=lambda: int(os.getenv("SAMPLE_SIZE", "0")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", "./data/cam_angle.db")))
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    concurrency: int = field(default_factory=lambda: int(os.getenv("CONCURRENCY", "4")))
    capture_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CAPTURE_CONCURRENCY", "8"))
    )
    inference_concurrency: int = field(
        default_factory=lambda: int(os.getenv("INFERENCE_CONCURRENCY", "24"))
    )
    capture_interval: float = field(default_factory=lambda: float(os.getenv("CAPTURE_INTERVAL", "5.0")))
    auth_session_hours: int = field(
        default_factory=lambda: int(os.getenv("AUTH_SESSION_HOURS", "12"))
    )
    auth_cookie_secure: bool = field(
        default_factory=lambda: _env_bool("AUTH_COOKIE_SECURE", False)
    )
    pipeline_mode: str = field(
        default_factory=lambda: os.getenv("PIPELINE_MODE", PIPELINE_MODE_DEFAULTS)
    )
    artifact_root: Optional[Path] = field(
        default_factory=lambda: (
            Path(value) if (value := os.getenv("ARTIFACT_ROOT", "").strip()) else None
        )
    )

    # Qwen 通过 OpenAI 兼容接口调用。模型服务和 Web 服务解耦，避免每次
    # 巡检都重新加载数 GB 权重。
    vlm_enabled: bool = field(default_factory=lambda: _env_bool("VLM_ENABLED", False))
    vlm_base_url: str = field(
        default_factory=lambda: os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    )
    vlm_model: str = field(
        default_factory=lambda: os.getenv("VLM_MODEL", "Qwen3-VL-4B-Instruct-AWQ-4bit")
    )
    vlm_api_key: str = field(default_factory=lambda: os.getenv("VLM_API_KEY", "EMPTY"))
    vlm_timeout: float = field(default_factory=lambda: float(os.getenv("VLM_TIMEOUT", "180")))
    vlm_max_image_side: int = field(
        default_factory=lambda: int(os.getenv("VLM_MAX_IMAGE_SIDE", "1280"))
    )
    vlm_jpeg_quality: int = field(
        default_factory=lambda: int(os.getenv("VLM_JPEG_QUALITY", "90"))
    )
    vlm_review_score: float = field(
        default_factory=lambda: float(os.getenv("VLM_REVIEW_SCORE", "0.09"))
    )
    vlm_min_element_confidence: float = field(
        default_factory=lambda: float(os.getenv("VLM_MIN_ELEMENT_CONFIDENCE", "0.55"))
    )
    vlm_large_occlusion_ratio: float = field(
        default_factory=lambda: float(os.getenv("VLM_LARGE_OCCLUSION_RATIO", "0.35"))
    )
    module_switches: Dict[str, bool] = field(
        default_factory=lambda: dict(MODULE_SWITCH_DEFAULTS)
    )
    vlm_prompts: Dict[str, str] = field(default_factory=dict)

    # 判定阈值，先给经验值，生产环境需要按真实数据标定。
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "dark_ratio": 0.95,
            "bright_ratio": 0.95,
            "mean_dark": 10.0,
            "mean_bright": 245.0,
            "blur_var": 60.0,
            "blur_region_ratio": 0.5,
            "blur_min_tiles": 4,
            "std_low": 18.0,
            "edge_density_low": 0.004,
            "roll_fail_deg": 20.0,
            "roll_review_deg": 8.0,
            "baseline_min_good": 8,
            "baseline_inlier_ratio": 0.35,
            "baseline_translation": 0.18,
            "baseline_rotation_deg": 18.0,
            "confidence_review": 0.78,
            "occlusion_area_ratio": 0.35,
            "occlusion_dark_value": 32.0,
            "occlusion_gray_saturation": 35.0,
            "occlusion_gray_max_value": 220.0,
            "occlusion_texture_std": 10.0,
            "occlusion_min_region_ratio": 0.005,
            "vlm_review_score": float(os.getenv("VLM_REVIEW_SCORE", "0.09")),
            "vlm_min_element_confidence": float(
                os.getenv("VLM_MIN_ELEMENT_CONFIDENCE", "0.55")
            ),
            "vlm_large_occlusion_ratio": float(
                os.getenv("VLM_LARGE_OCCLUSION_RATIO", "0.35")
            ),
        }
    )
    sop_rules: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "default": {"min_edge_density": 0.004, "min_std": 18.0, "max_roll_deg": 8.0},
            "indoor": {"min_edge_density": 0.005, "min_std": 18.0, "max_roll_deg": 6.0},
            "outdoor": {"min_edge_density": 0.003, "min_std": 15.0, "max_roll_deg": 5.0},
        }
    )

    def ensure_dirs(self) -> None:
        if self.artifact_root is None:
            self.artifact_root = self.data_dir / "artifacts"
        images_dir = self.data_dir / "images"
        thumbs_dir = self.data_dir / "thumbs"
        baselines_dir = self.data_dir / "baselines"
        for path in (self.data_dir, images_dir, thumbs_dir, baselines_dir, self.artifact_root):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def baselines_dir(self) -> Path:
        return self.data_dir / "baselines"


# OpenCV 的亮度、饱和度均为 0–255；面积比例使用 0–1。
_THRESHOLD_BOUNDS = {
    "dark_ratio": (0.01, 1.0),
    "bright_ratio": (0.01, 1.0),
    "mean_dark": (0.0, 254.0),
    "mean_bright": (1.0, 255.0),
    "blur_var": (0.01, 1000000.0),
    "blur_region_ratio": (0.01, 1.0),
    "blur_min_tiles": (1.0, 16.0),
    "std_low": (0.0, 128.0),
    "edge_density_low": (0.0, 1.0),
    "roll_fail_deg": (0.1, 90.0),
    "roll_review_deg": (0.0, 90.0),
    "baseline_min_good": (4.0, 2000.0),
    "baseline_inlier_ratio": (0.0, 1.0),
    "baseline_translation": (0.0, 2.0),
    "baseline_rotation_deg": (0.0, 180.0),
    "confidence_review": (0.0, 1.0),
    "occlusion_area_ratio": (0.01, 1.0),
    "occlusion_dark_value": (0.0, 254.0),
    "occlusion_gray_saturation": (0.0, 255.0),
    "occlusion_gray_max_value": (1.0, 255.0),
    "occlusion_texture_std": (0.0, 128.0),
    "occlusion_min_region_ratio": (0.000001, 0.25),
    "vlm_review_score": (0.0, 1.0),
    "vlm_min_element_confidence": (0.0, 1.0),
    "vlm_large_occlusion_ratio": (0.01, 1.0),
}
_THRESHOLD_PAIRS = (
    ("mean_dark", "mean_bright"),
    ("roll_review_deg", "roll_fail_deg"),
    ("occlusion_dark_value", "occlusion_gray_max_value"),
    ("occlusion_min_region_ratio", "occlusion_area_ratio"),
)


def validate_thresholds(current: Dict[str, float], updates: Dict[str, Any]) -> Dict[str, float]:
    """原子校验完整候选配置；失败时调用方的配置保持不变。"""
    candidate = dict(current)
    for key, value in updates.items():
        if key not in _THRESHOLD_BOUNDS:
            raise ValueError(f"未知判定阈值：{key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"阈值 {key} 必须为数字")
        low, high = _THRESHOLD_BOUNDS[key]
        if not low <= value <= high or not math.isfinite(value):
            raise ValueError(f"阈值 {key} 必须在 {low:g}–{high:g} 之间且为有限数字")
        if key in {"baseline_min_good", "blur_min_tiles"} and int(value) != value:
            raise ValueError(f"阈值 {key} 必须为整数")
        candidate[key] = float(value)
    for lower, upper in _THRESHOLD_PAIRS:
        if candidate[lower] >= candidate[upper]:
            raise ValueError(f"阈值 {lower} 必须小于 {upper}")
    return candidate


def restore_thresholds(defaults: Dict[str, float], saved: Any) -> Dict[str, float]:
    """兼容旧配置，并忽略越界、未知或相互冲突的历史设置。"""
    restored = dict(defaults)
    if not isinstance(saved, dict):
        return restored
    # 先逐项校验范围，再检查成对约束，避免加载顺序影响合法成组更新。
    for key, value in saved.items():
        if key not in _THRESHOLD_BOUNDS or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        low, high = _THRESHOLD_BOUNDS[key]
        if not low <= value <= high or not math.isfinite(value):
            continue
        if key in {"baseline_min_good", "blur_min_tiles"} and int(value) != value:
            continue
        restored[key] = float(value)
    for lower, upper in _THRESHOLD_PAIRS:
        if restored[lower] >= restored[upper]:
            restored[lower] = defaults[lower]
            restored[upper] = defaults[upper]
    return restored


def validate_module_switches(current: Dict[str, bool], updates: Any) -> Dict[str, bool]:
    if not isinstance(updates, dict):
        raise ValueError("模块开关必须是对象")
    candidate = {key: bool(current.get(key, True)) for key in MODULE_SWITCH_KEYS}
    for key, value in updates.items():
        if key not in MODULE_SWITCH_KEYS:
            raise ValueError(f"未知模块开关：{key}")
        if not isinstance(value, bool):
            raise ValueError(f"模块开关 {key} 必须是布尔值")
        candidate[key] = value
    return candidate


def restore_module_switches(saved: Any) -> Dict[str, bool]:
    restored = dict(MODULE_SWITCH_DEFAULTS)
    if not isinstance(saved, dict):
        return restored
    for key in MODULE_SWITCH_KEYS:
        if isinstance(saved.get(key), bool):
            restored[key] = saved[key]
    return restored


def validate_pipeline_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in PIPELINE_MODES:
        raise ValueError("检测链路必须是 coupled 或 decoupled")
    return mode




_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_dirs()
    return _config
