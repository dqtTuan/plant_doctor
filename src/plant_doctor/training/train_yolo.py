"""Bước 1: Tự huấn luyện YOLOv8n trên dataset bệnh lá cây.

python -m plant_doctor.training.train_yolo                # Đọc trực tiếp từ Drive/ổ cứng (cache=False)
python -m plant_doctor.training.train_yolo --batch 16    # 100 epochs, batch 16
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from plant_doctor.config import DATASET_DIR, SEED, YOLO_BASE, YOLO_NAME, YOLO_PROJECT, YOLO_WEIGHTS


def normalize_data_yaml(yaml_path: Path | str) -> Path:
    """Tự động kiểm tra và chuẩn hóa cấu hình data.yaml (sửa lỗi xuất relative paths kiểu ../ của Roboflow)."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return yaml_path

    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return yaml_path

        changed = False
        base_dir = yaml_path.parent

        # Chuẩn hóa path về thư mục chứa data.yaml nếu chưa có hoặc không trỏ tới thư mục tồn tại
        current_path = cfg.get("path")
        if not current_path or not Path(current_path).is_absolute() or not Path(current_path).exists():
            cfg["path"] = str(base_dir.resolve()).replace("\\", "/")
            changed = True

        root_path = Path(cfg["path"])
        for key in ["train", "val", "test"]:
            val = cfg.get(key)
            if not isinstance(val, str):
                continue
            # Bỏ tiền tố ../ nếu thư mục thực tế nằm ngay trong root_path
            if val.startswith("../"):
                stripped = val[3:]
                if (root_path / stripped).exists():
                    cfg[key] = stripped
                    changed = True

        if changed:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            print(f"Đã tự động chuẩn hóa đường dẫn trong {yaml_path}")
    except Exception as e:
        print(f"Cảnh báo khi kiểm tra {yaml_path}: {e}")

    return yaml_path


def train_yolo(data_yaml: Path | str = DATASET_DIR / "data.yaml",
               model_weights: Path | str = YOLO_BASE,
               epochs: int = 100,
               batch: int = 16,
               imgsz: int = 640,
               freeze: int = 0,
               project: Path | str = YOLO_PROJECT,
               name: str = YOLO_NAME,
               device: str | int | None = None,
               cache: bool | str = False,
               workers: int = 2,
               amp: bool = True) -> dict:
    """Huấn luyện mô hình YOLOv8 trên tập dữ liệu và đánh giá trên tập test."""
    data_yaml = normalize_data_yaml(data_yaml)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # Chuẩn hóa giá trị cache (mặc định False: đọc trực tiếp từ Drive/ổ cứng, không tốn RAM)
    if isinstance(cache, str):
        if cache.lower() in {"false", "none", "0"}:
            cache = False
        elif cache.lower() == "true":
            cache = True

    model = YOLO(str(model_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache=cache,
        workers=workers,
        amp=amp,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,
        fliplr=0.5,
        freeze=freeze if freeze > 0 else None,
        seed=SEED,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    best = YOLO(str(YOLO_WEIGHTS))
    metrics = best.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device)
    print(f"Test mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")
    print(f"Trọng số: {YOLO_WEIGHTS}")
    return {"map50": float(metrics.box.map50), "map": float(metrics.box.map)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--model", default=str(YOLO_BASE))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16,
                    help="Kích thước batch (16 hoặc 32)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--freeze", type=int, default=0,
                    help="Số layer đầu bị đóng băng (0 = train toàn bộ mạng)")
    ap.add_argument("--device", default=None,
                    help="Thiết bị tính toán (0, 1, 'cpu' hoặc None để tự động chọn GPU)")
    ap.add_argument("--cache", default="false",
                    help="Bộ đệm dữ liệu ('false' để đọc trực tiếp từ Drive/ổ cứng, không tốn RAM)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Số luồng đọc dữ liệu song song (mặc định 2 phù hợp trực tiếp với Drive)")
    args = ap.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_weights=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        device=args.device,
        cache=args.cache,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
