"""Chia tập dữ liệu có thể tái lập, theo nhóm để tránh rò rỉ dữ liệu (data leakage).

Các mẫu cùng nhóm (VD: cùng một đoạn tài liệu / cùng một bệnh) luôn nằm chung một tập,
nên điểm trên tập test phản ánh khả năng tổng quát hoá thay vì ghi nhớ.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def _bucket(key: str, seed: int) -> float:
    """Ánh xạ ổn định key -> [0, 1) (không phụ thuộc thứ tự, PYTHONHASHSEED hay máy chạy)."""
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def group_split(items: Sequence[T], group_fn: Callable[[T], str], val: float = 0.1, test: float = 0.1,
                seed: int = 42) -> dict[str, list[T]]:
    """Chia items thành train/val/test theo nhóm. Đảm bảo val/test không rỗng khi đủ nhóm."""
    if val < 0 or test < 0 or val + test >= 1:
        raise ValueError("Tỉ lệ val/test không hợp lệ")
    groups = sorted({group_fn(it) for it in items}, key=lambda g: _bucket(g, seed))
    n = len(groups)
    n_test = round(n * test)
    n_val = round(n * val)
    if n >= 3:  # tập nhỏ: vẫn giữ tối thiểu 1 nhóm cho mỗi tập được yêu cầu
        n_test = max(n_test, 1 if test > 0 else 0)
        n_val = max(n_val, 1 if val > 0 else 0)
    assign = {g: "test" for g in groups[:n_test]}
    assign.update({g: "val" for g in groups[n_test:n_test + n_val]})
    out: dict[str, list[T]] = {"train": [], "val": [], "test": []}
    for it in items:
        out[assign.get(group_fn(it), "train")].append(it)
    return out

