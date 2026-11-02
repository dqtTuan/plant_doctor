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
KB_DIR = DATA_DIR / "knowledge_base"        # tài liệu PDF/MD/TXT chuyên sâu nông nghiệp

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


# --- Tối ưu hóa suy luận & Truy xuất lai ---
DEFAULT_MAX_CONTEXT_DOCS = 4
DEFAULT_MAX_NEW_TOKENS = 256
RETRIEVE_CANDIDATES = 20
RRF_K = 60
DENSE_MIN_SCORE = 0.35
RERANK_MIN_SCORE = 0.0


# --- PostgreSQL (pgvector) ---
PGVECTOR_URL = os.environ.get("PGVECTOR_URL", "")
PGVECTOR_TABLE = "plant_doctor_embeddings"


def embed_max_tokens(embed_dir: Path | str = EMBED_DIR) -> int:
    """Đọc giới hạn độ dài token từ cấu hình mô hình embedding (mặc định 126)."""
    cfg_file = Path(embed_dir) / "sentence_bert_config.json"
    if cfg_file.exists():
        import json
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if "max_seq_length" in cfg:
                return max(16, int(cfg["max_seq_length"]) - 2)
        except Exception:
            pass
    return 126
