"""Nạp cấu hình thí nghiệm từ configs/<name>.yaml và ghi đè bằng tham số dòng lệnh.

    cfg = load_config("yolo", ["train.epochs=50", "train.batch=32"])
    cfg["train"]["epochs"]  # -> 50

Giá trị ghi đè được phân tích bằng YAML nên `true`, `0.1`, `[a, b]`, `null` đều đúng kiểu.
Chuỗi `${ROOT}` trong file YAML được thay bằng thư mục gốc dự án.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from plant_doctor.config.settings import CONFIGS_DIR, ROOT


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("${ROOT}", str(ROOT).replace("\\", "/"))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Áp dụng danh sách `a.b.c=value` lên bản sao của cfg."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Ghi đè không hợp lệ '{item}', cần dạng key=value (VD train.epochs=50)")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Không thể ghi đè '{key}': '{p}' không phải nhóm cấu hình")
        node[parts[-1]] = yaml.safe_load(raw) if raw != "" else ""
    return cfg


def load_config(name_or_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Đọc configs/<name>.yaml (hoặc đường dẫn file) rồi áp dụng ghi đè."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand(apply_overrides(cfg, overrides))

