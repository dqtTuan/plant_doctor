"""Bước 0: Chuyển dataset ảnh chỉ có nhãn phân loại (kiểu PlantVillage) sang định dạng YOLO detection.

Đầu vào:  <src>/<ten_lop>/*.jpg                 (mặc định data/raw/)
Đầu ra:   <out>/images/{train,val,test}, <out>/labels/{train,val,test}, <out>/data.yaml (mặc định data/yolo/)

Vì ảnh không có bounding box, khung được sinh tự động bằng cách khoanh vùng phiến lá:
  --method opencv : tách lá theo kênh bão hòa màu (Otsu) – nhanh, hợp với nền đồng màu như PlantVillage
  --method gdino  : GroundingDINO zero-shot ("a leaf.") – tốt hơn với ảnh ngoài đồng ruộng, cần GPU
Nếu không tìm được lá, khung = toàn bộ ảnh.

Ví dụ:
  python -m plant_doctor.data.prepare_dataset --src data/raw --method opencv --max-per-class 400
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from plant_doctor.config import DATASET_DIR, RAW_DATA_DIR, SEED

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ------------------------------------------------------------ Sinh bounding box
def bbox_opencv(img: np.ndarray):
    h, w = img.shape[:2]
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, mask = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * h * w:  # quá nhỏ -> nhiều khả năng là nhiễu
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    return x, y, x + bw, y + bh


class GroundingDinoBoxer:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", prompt="a leaf."):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.prompt = prompt

    def __call__(self, img: np.ndarray):
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        kwargs = dict(text_threshold=0.25, target_sizes=[pil.size[::-1]])
        try:  # tên tham số thay đổi giữa các phiên bản transformers
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=0.3, **kwargs)[0]
        except TypeError:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.3, **kwargs)[0]
        if len(res["scores"]) == 0:
            return None
        x1, y1, x2, y2 = res["boxes"][int(res["scores"].argmax())].tolist()
        return int(x1), int(y1), int(x2), int(y2)


# ------------------------------------------------------------ Tiện ích
def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=RAW_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--method", choices=["opencv", "gdino"], default="opencv")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = dùng toàn bộ ảnh")
    args = ap.parse_args()

    class_dirs = sorted(d for d in args.src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"Không tìm thấy thư mục lớp nào trong {args.src}")
    names = [d.name for d in class_dirs]
    print(f"Tìm thấy {len(names)} lớp: {names}")

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)

    boxer = GroundingDinoBoxer() if args.method == "gdino" else bbox_opencv
    rng = random.Random(SEED)
    stats = {s: 0 for s in SPLITS}
    fallback = 0

    for cls_id, cdir in enumerate(class_dirs):
        files = sorted(p for p in cdir.rglob("*") if p.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        if args.max_per_class:
            files = files[: args.max_per_class]
        n_train = int(len(files) * SPLITS["train"])
        n_val = int(len(files) * SPLITS["val"])
        split_of = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(files) - n_train - n_val)

        for i, (path, split) in enumerate(tqdm(list(zip(files, split_of)), desc=cdir.name)):
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = resize_max(img, args.imgsz)
            h, w = img.shape[:2]
            box = boxer(img)
            if box is None:
                box, fallback = (0, 0, w, h), fallback + 1
            stem = f"{cls_id:03d}_{i:06d}"
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), img)
            cx, cy, bw, bh = to_yolo(box, w, h)
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats[split] += 1

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    with open(args.out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"Xong: {stats} | ảnh dùng khung toàn ảnh (không tách được lá): {fallback}")
    print(f"data.yaml: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
