"""Bước 1: Tự huấn luyện YOLOv8n trên dataset bệnh lá cây.

python -m plant_doctor.training.train_yolo                # 100 epochs, batch 16 (theo thiết kế)
python -m plant_doctor.training.train_yolo --epochs 30    # thử nhanh trên Colab
python -m plant_doctor.training.train_yolo --freeze 0     # không đóng băng, train toàn bộ mạng
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from plant_doctor.config import DATASET_DIR, SEED, YOLO_BASE, YOLO_NAME, YOLO_PROJECT, YOLO_WEIGHTS


def train_yolo(data_yaml: Path | str = DATASET_DIR / "data.yaml",
               model_weights: Path | str = YOLO_BASE,
               epochs: int = 100,
               batch: int = 16,
               imgsz: int = 640,
               freeze: int = 10,
               project: Path | str = YOLO_PROJECT,
               name: str = YOLO_NAME) -> dict:
    """Huấn luyện mô hình YOLOv8 trên tập dữ liệu và đánh giá trên tập test."""
    model = YOLO(str(model_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
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
    metrics = best.val(data=str(data_yaml), split="test", imgsz=imgsz)
    print(f"Test mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")
    print(f"Trọng số: {YOLO_WEIGHTS}")
    return {"map50": float(metrics.box.map50), "map": float(metrics.box.map)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--model", default=str(YOLO_BASE))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--freeze", type=int, default=10,
                    help="Số layer đầu bị đóng băng (10 = backbone YOLOv8, 0 = không đóng băng)")
    args = ap.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_weights=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
    )


if __name__ == "__main__":
    main()
