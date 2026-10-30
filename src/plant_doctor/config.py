"""Cấu hình đường dẫn & tên mô hình dùng chung cho toàn bộ pipeline.

Thư mục gốc dự án mặc định là thư mục chứa `pyproject.toml` (2 cấp trên `src/plant_doctor`).
Có thể ghi đè bằng biến môi trường `PLANT_DOCTOR_ROOT` (VD: khi chạy trên Colab/Drive).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PLANT_DOCTOR_ROOT", Path(__file__).resolve().parents[2])).resolve()

# --- Dữ liệu ---
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"             # data/raw/<ten_lop>/*.jpg (nhãn phân loại kiểu PlantVillage)
DATASET_DIR = DATA_DIR / "yolo"             # dataset YOLO (sinh bởi bước 0 hoặc tải từ Roboflow)
KB_DIR = DATA_DIR / "knowledge_base"        # diseases.json + các file PDF/MD/TXT bạn tự biên soạn
KB_JSON = KB_DIR / "diseases.json"
TRAIN_DATA_DIR = DATA_DIR / "training"      # dữ liệu bổ sung (tuỳ chọn) cho embedding & LLM

# --- Mô hình ---
MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
YOLO_BASE = PRETRAINED_DIR / "yolov8n.pt"   # trọng số khởi tạo (tự tải nếu chưa có)
YOLO_PROJECT = MODELS_DIR
YOLO_NAME = "yolo_leaf_disease"
YOLO_WEIGHTS = MODELS_DIR / YOLO_NAME / "weights" / "best.pt"

# Mô hình nhúng gốc: bản đa ngôn ngữ (all-MiniLM-L6-v2 chỉ hiểu tiếng Anh)
EMBED_BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIR = MODELS_DIR / "custom_embedding"

# Mô hình ngôn ngữ nhỏ gốc để fine-tune LoRA
LLM_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_ADAPTER_DIR = MODELS_DIR / "custom_llm_lora_adapter"   # chỉ trọng số LoRA
LLM_MERGED_DIR = MODELS_DIR / "custom_llm_lora"            # mô hình đã gộp LoRA, dùng khi suy luận
LLM_CHECKPOINT_DIR = MODELS_DIR / "llm_lora_checkpoints"   # checkpoint tạm khi train

# --- Vector DB ---
FAISS_DIR = ROOT / "vector_store" / "faiss_index"

SEED = 42


# --- Tối ưu hóa suy luận (Inference Latency) ---
DEFAULT_MAX_CONTEXT_DOCS = 2
DEFAULT_MAX_NEW_TOKENS = 256
