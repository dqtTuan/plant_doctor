"""Theo dõi thí nghiệm cục bộ (offline): mỗi lần chạy = một thư mục runs/<task>/<timestamp>/.

    run = Run.create("yolo", cfg)        # ghi config.yaml + env.json
    run.log_metrics({"map50": 0.91})     # ghi/merge metrics.json
    run.write_report("# Kết quả ...")    # report.md
    run.finish()                          # cập nhật runs/<task>/latest.json + runs/summary.csv

Cấu trúc thư mục:
    runs/
    ├── summary.csv                  # mỗi dòng = 1 lần chạy (task, run, metrics chính) – so sánh nhanh
    └── yolo/
        ├── latest.json              # trỏ tới lần chạy gần nhất
        └── 20261106-221500/
            ├── config.yaml          # cấu hình đã áp dụng ghi đè
            ├── env.json             # git commit, phiên bản thư viện, GPU
            ├── metrics.json
            └── report.md
"""
from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from plant_doctor.config.settings import ROOT, RUNS_DIR


def set_seed(seed: int) -> None:
    """Cố định seed cho random / numpy / torch (nếu có) để tái lập kết quả."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) or None
    except Exception:
        return None


def environment_info() -> dict:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                            "git_commit": _git_commit()}
    for pkg in ("torch", "ultralytics", "transformers", "sentence_transformers", "peft"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy / torch scalar
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


@dataclass
class Run:
    task: str
    dir: Path
    metrics: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    @classmethod
    def create(cls, task: str, cfg: dict | None = None, root: Path | str = RUNS_DIR,
               name: str | None = None) -> Run:
        name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(root) / task / name
        i = 1
        while run_dir.exists():  # tránh trùng khi chạy 2 lần trong cùng giây
            run_dir = Path(root) / task / f"{name}-{i}"
            i += 1
        run_dir.mkdir(parents=True)
        run = cls(task=task, dir=run_dir)
        if cfg is not None:
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(_to_jsonable(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
        run.save_json("env.json", environment_info())
        print(f"📁 Thư mục thí nghiệm: {run_dir}")
        return run

    def save_json(self, filename: str, data: Any) -> Path:
        path = self.dir / filename
        path.write_text(json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def log_metrics(self, metrics: dict, prefix: str | None = None) -> None:
        items = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        self.metrics.update(_to_jsonable(items))
        self.save_json("metrics.json", self.metrics)

    def write_report(self, markdown: str) -> Path:
        path = self.dir / "report.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def finish(self) -> None:
        duration = round(time.time() - self.started, 1)
        self.log_metrics({"duration_sec": duration})
        root = self.dir.parent.parent
        (self.dir.parent / "latest.json").write_text(
            json.dumps({"run": self.dir.name, "path": str(self.dir)}, ensure_ascii=False), encoding="utf-8")
        scalars = {k: v for k, v in self.metrics.items() if isinstance(v, (int, float, str, bool))}
        _append_summary(root / "summary.csv", {"task": self.task, "run": self.dir.name, **scalars})
        print(f"✅ Đã lưu kết quả: {self.dir / 'metrics.json'}")


def _append_summary(path: Path, row: dict) -> None:
    """Ghi thêm 1 dòng vào summary.csv, tự mở rộng cột khi xuất hiện chỉ số mới."""
    rows: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append({k: str(v) for k, v in row.items()})
    columns: list[str] = []
    for r in rows:
        columns += [c for c in r if c not in columns]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metrics_table(metrics: dict, title: str = "Chỉ số") -> str:
    """Bảng Markdown cho các chỉ số dạng số (dùng trong report.md)."""
    lines = [f"| {title} | Giá trị |", "|---|---|"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        elif isinstance(v, (int, str, bool)):
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

