from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    capture_interval: float = field(default_factory=lambda: float(os.getenv("CAPTURE_INTERVAL", "5.0")))

    # 判定阈值，先给经验值，生产环境需要按真实数据标定。
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "dark_ratio": 0.95,
            "bright_ratio": 0.95,
            "mean_dark": 10.0,
            "mean_bright": 245.0,
            "blur_var": 60.0,
            "std_low": 18.0,
            "edge_density_low": 0.004,
            "roll_fail_deg": 20.0,
            "roll_review_deg": 8.0,
            "baseline_min_good": 8,
            "baseline_inlier_ratio": 0.35,
            "baseline_translation": 0.18,
            "baseline_rotation_deg": 18.0,
            "confidence_review": 0.78,
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
        images_dir = self.data_dir / "images"
        thumbs_dir = self.data_dir / "thumbs"
        baselines_dir = self.data_dir / "baselines"
        for path in (self.data_dir, images_dir, thumbs_dir, baselines_dir):
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


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_dirs()
    return _config
