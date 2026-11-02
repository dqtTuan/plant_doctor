"""CLI tải mô hình Cross-Encoder Reranker về thư mục models/reranker/ (chạy 1 lần khi có internet)."""
import os
from pathlib import Path

from sentence_transformers import CrossEncoder

from plant_doctor.config import RERANKER_BASE, RERANKER_DIR


def main():
    print(f"Đang tải mô hình Cross-Encoder: {RERANKER_BASE}...")
    # Tạm thời mở kết nối để tải trọng số
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    out_path = Path(RERANKER_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(RERANKER_BASE)
    model.save(str(out_path))
    print(f"Đã lưu mô hình Reranker tại: {out_path}")


if __name__ == "__main__":
    main()
