"""Bước 1: Tự huấn luyện YOLOv8n trên dataset bệnh lá cây.

python -m plant_doctor.training.train_yolo                # 100 epochs, batch 16 (theo thiết kế)
python -m plant_doctor.training.train_yolo --epochs 30    # thử nhanh trên Colab
python -m plant_doctor.training.train_yolo --freeze 0     # không đóng băng, train toàn bộ mạng
"""
import argparse

from ultralytics import YOLO

from plant_doctor.config import DATASET_DIR, SEED, YOLO_BASE, YOLO_NAME, YOLO_PROJECT, YOLO_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--model", default=str(YOLO_BASE))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    # YOLOv8: layer 0-9 là backbone (CSPDarknet + SPPF), 10-21 là neck + head.
    # --freeze 10 = đóng băng toàn bộ backbone, chỉ train neck + head (nhanh, ít overfit
    # khi dataset nhỏ); --freeze 0 = train toàn bộ.
    ap.add_argument("--freeze", type=int, default=10,
                    help="Số layer đầu bị đóng băng (10 = backbone YOLOv8, 0 = không đóng băng)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        mosaic=1.0,
        fliplr=0.5,
        freeze=args.freeze if args.freeze > 0 else None,
        seed=SEED,
        project=str(YOLO_PROJECT),  # đường dẫn tuyệt đối để không bị lồng vào runs/detect
        name=YOLO_NAME,
        exist_ok=True,
    )

    # Đánh giá khách quan trên tập test (15%) chưa từng dùng khi huấn luyện
    best = YOLO(str(YOLO_WEIGHTS))
    metrics = best.val(data=args.data, split="test", imgsz=args.imgsz)
    print(f"Test mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")
    print(f"Trọng số: {YOLO_WEIGHTS}")


if __name__ == "__main__":
    main()
