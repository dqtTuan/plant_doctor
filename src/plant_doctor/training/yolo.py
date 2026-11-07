"""Huấn luyện YOLOv8 phát hiện bệnh lá + đánh giá trên tập test.

    pd train yolo                                   # cấu hình: configs/yolo.yaml
    pd train yolo train.epochs=30 train.batch=32

Kết quả:
    runs/yolo/<timestamp>/train/       log & biểu đồ của Ultralytics (weights/best.pt, results.csv, ...)
    runs/yolo/<timestamp>/eval_test/   confusion matrix, đường cong PR trên tập test
    runs/yolo/<timestamp>/metrics.json, report.md
    models/yolo_leaf_disease/weights/best.pt   bản phát hành dùng cho app (model.output)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from plant_doctor.utils.experiment import Run, set_seed


def normalize_data_yaml(yaml_path: Path | str) -> Path:
    """Chuẩn hóa data.yaml (sửa đường dẫn tương đối kiểu ../ của Roboflow, điền `path`)."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return yaml_path
    try:
        import yaml
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            return yaml_path

        changed = False
        current_path = cfg.get("path")
        if not current_path or not Path(current_path).is_absolute() or not Path(current_path).exists():
            cfg["path"] = str(yaml_path.parent.resolve()).replace("\\", "/")
            changed = True

        root_path = Path(cfg["path"])
        for key in ("train", "val", "test"):
            val = cfg.get(key)
            if isinstance(val, str) and val.startswith("../") and (root_path / val[3:]).exists():
                cfg[key] = val[3:]
                changed = True

        if changed:
            yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"Đã tự động chuẩn hóa đường dẫn trong {yaml_path}")
    except Exception as e:
        print(f"Cảnh báo khi kiểm tra {yaml_path}: {e}")
    return yaml_path


def _resolve_device(device):
    import torch
    if device is None:
        return 0 if torch.cuda.is_available() else "cpu"
    return device


def train_yolo(cfg: dict) -> dict:
    """Huấn luyện theo cfg (configs/yolo.yaml), đánh giá trên tập test, trả về metrics."""
    import torch
    from ultralytics import YOLO

    from plant_doctor.evaluation.yolo import evaluate_yolo, yolo_report

    set_seed(cfg["seed"])
    t, ev = cfg["train"], cfg["eval"]
    data_yaml = normalize_data_yaml(cfg["data"]["yaml"])
    device = _resolve_device(t.get("device"))
    gpu = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"🚀 Thiết bị huấn luyện YOLO: {gpu} | batch={t['batch']} | epochs={t['epochs']}")

    run = Run.create("yolo", cfg)
    model = YOLO(str(cfg["model"]["weights"]))
    model.train(
        data=str(data_yaml), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"], device=device,
        cache=t.get("cache", False), workers=t.get("workers", 2), amp=t.get("amp", True),
        patience=t.get("patience", 100), optimizer=t.get("optimizer", "SGD"), lr0=t["lr0"],
        momentum=t["momentum"], weight_decay=t["weight_decay"], mosaic=t["mosaic"], fliplr=t["fliplr"],
        freeze=t["freeze"] if t.get("freeze") else None, seed=cfg["seed"],
        project=str(run.dir), name="train", exist_ok=True,
    )

    best = run.dir / "train" / "weights" / "best.pt"
    output = Path(cfg["model"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"Trọng số phát hành: {output}")

    result = evaluate_yolo(best, data_yaml, split=ev["split"], imgsz=t["imgsz"], conf=ev["conf"],
                           iou=ev["iou"], device=device, save_dir=run.dir / f"eval_{ev['split']}")
    run.log_metrics(result["overall"], prefix=ev["split"])
    run.save_json("per_class.json", result["per_class"])
    run.write_report(yolo_report(result, ev["split"]))
    run.finish()
    o = result["overall"]
    print(f"[{ev['split']}] P={o['precision']:.4f} R={o['recall']:.4f} mAP50={o['map50']:.4f} "
          f"mAP50-95={o['map50_95']:.4f}")
    return result["overall"]


def evaluate_yolo_run(cfg: dict) -> dict:
    """Chỉ đánh giá (không train) trọng số model.output trên eval.split."""
    from plant_doctor.evaluation.yolo import evaluate_yolo, yolo_report

    ev = cfg["eval"]
    run = Run.create("yolo_eval", cfg)
    data_yaml = normalize_data_yaml(cfg["data"]["yaml"])
    result = evaluate_yolo(cfg["model"]["output"], data_yaml, split=ev["split"], imgsz=cfg["train"]["imgsz"],
                           conf=ev["conf"], iou=ev["iou"], device=_resolve_device(cfg["train"].get("device")),
                           save_dir=run.dir / f"eval_{ev['split']}")
    run.log_metrics(result["overall"], prefix=ev["split"])
    run.save_json("per_class.json", result["per_class"])
    run.write_report(yolo_report(result, ev["split"]))
    run.finish()
    print(yolo_report(result, ev["split"]))
    return result["overall"]
