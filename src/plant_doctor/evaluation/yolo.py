"""Đánh giá YOLOv8 trên một tập (mặc định test): precision, recall, F1, mAP50, mAP50-95,
chỉ số theo từng lớp, tốc độ suy luận; lưu confusion matrix & đường cong PR vào thư mục run.

    pd eval yolo                                  # dùng models/yolo_leaf_disease/weights/best.pt
    pd eval yolo model.output=path/to/best.pt eval.split=val
"""
from __future__ import annotations

from pathlib import Path


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_yolo(weights: Path | str, data_yaml: Path | str, split: str = "test", imgsz: int = 640,
                  conf: float = 0.001, iou: float = 0.6, device=None,
                  save_dir: Path | str | None = None) -> dict:
    """Trả về {"overall": {...}, "per_class": [{class, precision, recall, f1, map50, map50_95}], "speed_ms": {...}}."""
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    kwargs = dict(data=str(data_yaml), split=split, imgsz=imgsz, conf=conf, iou=iou, device=device,
                  plots=save_dir is not None, verbose=False)
    if save_dir is not None:
        save_dir = Path(save_dir)
        kwargs.update(project=str(save_dir.parent), name=save_dir.name, exist_ok=True)

    metrics = YOLO(str(weights)).val(**kwargs)
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    overall = {"precision": p, "recall": r, "f1": _f1(p, r),
               "map50": float(box.map50), "map50_95": float(box.map), "map75": float(box.map75)}

    per_class = []
    for i, cls_idx in enumerate(box.ap_class_index):
        cp, cr, ap50, ap = (float(x) for x in box.class_result(i))
        per_class.append({"class": metrics.names[int(cls_idx)], "precision": cp, "recall": cr,
                          "f1": _f1(cp, cr), "map50": ap50, "map50_95": ap})
    speed = {k: round(float(v), 2) for k, v in (getattr(metrics, "speed", {}) or {}).items()}
    return {"overall": overall, "per_class": per_class, "speed_ms": speed}


def yolo_report(result: dict, split: str) -> str:
    o = result["overall"]
    lines = [f"# Đánh giá YOLOv8 – tập `{split}`", "",
             "| Precision | Recall | F1 | mAP50 | mAP50-95 |", "|---|---|---|---|---|",
             f"| {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} | {o['map50']:.4f} | {o['map50_95']:.4f} |",
             "", "## Theo từng lớp", "", "| Lớp | Precision | Recall | F1 | mAP50 | mAP50-95 |",
             "|---|---|---|---|---|---|"]
    for c in sorted(result["per_class"], key=lambda c: c["map50"]):
        lines.append(f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} "
                     f"| {c['map50']:.3f} | {c['map50_95']:.3f} |")
    if result.get("speed_ms"):
        lines += ["", "Tốc độ (ms/ảnh): " + ", ".join(f"{k}={v}" for k, v in result["speed_ms"].items())]
    lines += ["", "Biểu đồ: `confusion_matrix*.png`, `*PR_curve.png`, `*F1_curve.png` trong thư mục này."]
    return "\n".join(lines) + "\n"

