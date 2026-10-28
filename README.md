# 🌿 Plant Doctor – YOLOv8 + Local RAG tự huấn luyện (100% Offline)

Hệ thống chẩn đoán bệnh lá cây theo kiến trúc Neuro-Symbolic: **YOLOv8n** phát hiện bệnh →
**FAISS + Embedding tự tinh chỉnh** truy xuất tri thức → **Qwen2.5-1.5B + LoRA** sinh phác đồ điều trị sinh học.

## Cấu trúc

```text
plant_doctor/
├── app/
│   └── streamlit_app.py            # Bước 5: giao diện web offline
├── data/                           # (gitignore, trừ knowledge_base/)
│   ├── raw/<ten_lop>/*.jpg         # ảnh có nhãn phân loại (PlantVillage...) – đầu vào bước 0
│   ├── yolo/                       # dataset YOLO (data.yaml + train/valid/test)
│   ├── knowledge_base/
│   │   ├── diseases.json           # tri thức có cấu trúc: triệu chứng → nguyên nhân → điều trị → phòng ngừa
│   │   └── *.pdf / *.md / *.txt    # (tuỳ chọn) tài liệu bạn tự biên soạn
│   └── training/                   # (tuỳ chọn) embedding_pairs.jsonl, llm_extra.jsonl
├── models/                         # (gitignore)
│   ├── pretrained/yolov8n.pt       # trọng số khởi tạo YOLO
│   ├── yolo_leaf_disease/          # kết quả bước 1 (weights/best.pt)
│   ├── custom_embedding/           # kết quả bước 2
│   └── custom_llm_lora/            # kết quả bước 3 (đã gộp LoRA)
├── vector_store/faiss_index/       # kết quả bước 4 (gitignore)
├── notebooks/
│   └── colab_pipeline.ipynb        # chạy toàn bộ bước 0-5 trên Colab
├── src/plant_doctor/               # package Python
│   ├── config.py                   # toàn bộ đường dẫn & tên mô hình
│   ├── knowledge.py                # tra cứu & định dạng kho tri thức
│   ├── prompts.py                  # system prompt + build_messages (dùng chung train/suy luận)
│   ├── data/prepare_dataset.py     # Bước 0: phân loại → YOLO detection, tự sinh bounding box
│   ├── training/
│   │   ├── train_yolo.py           # Bước 1
│   │   ├── train_embedding.py      # Bước 2
│   │   └── train_llm.py            # Bước 3
│   └── rag/build_index.py          # Bước 4
├── tests/                          # pytest (không cần GPU/mô hình)
├── .github/workflows/              # ci.yml, release.yml
├── Dockerfile · .dockerignore
├── docker-compose.yml · docker-compose.gpu.yml
├── pyproject.toml
└── requirements.txt
```

## Cài đặt

```bash
pip install torch  # theo hướng dẫn pytorch.org (bản CUDA nếu có GPU)
pip install -r requirements.txt   # cài thư viện + package plant_doctor (editable)
```

## Pipeline

| Bước | Lệnh | Lệnh tắt |
|---|---|---|
| 0 | `python -m plant_doctor.data.prepare_dataset --src data/raw` | `pd-prepare-dataset` |
| 1 | `python -m plant_doctor.training.train_yolo` | `pd-train-yolo` |
| 2 | `python -m plant_doctor.training.train_embedding` | `pd-train-embedding` |
| 3 | `python -m plant_doctor.training.train_llm` | `pd-train-llm` |
| 4 | `python -m plant_doctor.rag.build_index` | `pd-build-index` |
| 5 | `streamlit run app/streamlit_app.py` | |

Kiểm thử: `pip install -e .[dev] && pytest`

> Thư mục gốc được tự xác định từ vị trí package; có thể ghi đè bằng biến môi trường `PLANT_DOCTOR_ROOT`.

## Chạy trực tiếp trên Google Drive qua Colab

1. Tải thư mục dự án này lên Google Drive theo đường dẫn: `MyDrive/plant_doctor/`.
2. Đặt dữ liệu ảnh vào `MyDrive/plant_doctor/data/raw/<ten_lop>/*.jpg`.
3. Mở file `notebooks/colab_pipeline.ipynb` trên Colab, chọn Runtime GPU (T4), chạy lần lượt các ô.
4. Quá trình huấn luyện đọc dữ liệu và lưu thẳng toàn bộ kết quả (`models/`, `vector_store/`) vào Drive, không cần copy thủ công. Nếu Colab bị ngắt kết nối thì mô hình vẫn an toàn trên Drive.

## Chạy app offline trên máy cá nhân

```bash
# chép models/ và vector_store/ từ Drive về thư mục dự án
streamlit run app/streamlit_app.py
```

App tự đặt `HF_HUB_OFFLINE=1`, không kết nối mạng. Nếu thiếu LLM, app vẫn trả lời bằng tri thức
có cấu trúc trong `diseases.json`; nếu thiếu FAISS, app tra cứu trực tiếp theo nhãn YOLO.

## Docker

Image chỉ chứa code + thư viện + kho tri thức; **`models/` và `vector_store/` được mount từ máy chủ**.

```bash
# Build & chạy local
docker compose up -d --build                                              # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU (cần NVIDIA Container Toolkit)

# Hoặc dùng image đã build sẵn trên GHCR
export IMAGE=ghcr.io/<owner>/<repo>      # PowerShell: $env:IMAGE="ghcr.io/<owner>/<repo>"
docker compose pull && docker compose up -d
```

Mở http://localhost:8501. Đổi cổng bằng biến `PORT`, đổi phiên bản bằng `TAG` (VD `TAG=1.2.0`).

## CI/CD (GitHub Actions)

| Workflow | Kích hoạt | Việc làm |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | Pull Request, push `main` | ruff, pytest, kiểm tra `diseases.json`, hadolint, build image CPU + smoke test health check |
| [`release.yml`](.github/workflows/release.yml) | push `main`, tag `v*.*.*` | test → build & push image CPU + GPU lên `ghcr.io` (kèm provenance attestation); tag `v*` tạo GitHub Release |

Tag image: `latest` / `latest-gpu` (nhánh main), `1.2.3` / `1.2.3-gpu`, `1.2`, `sha-<commit>`.
Phát hành phiên bản mới: `git tag v0.1.0 && git push origin v0.1.0`.

> [!NOTE]
> Không cần cấu hình secret – workflow dùng `GITHUB_TOKEN`. Lần đầu push, vào **Packages** trên GitHub
> để đặt image ở chế độ public nếu muốn `docker pull` không cần đăng nhập.

## Khác biệt so với bản thiết kế ban đầu (và lý do)

| Thiết kế gốc | Bản triển khai | Lý do |
|---|---|---|
| Dataset có sẵn bounding box | `prepare_dataset.py` tự sinh khung bao phiến lá (OpenCV / GroundingDINO) | Dataset chỉ có nhãn phân loại. YOLO sẽ khoanh **cả lá** kèm tên bệnh, không khoanh từng vết bệnh |
| `all-MiniLM-L6-v2` + CosineSimilarityLoss | `paraphrase-multilingual-MiniLM-L12-v2` + MultipleNegativesRankingLoss | Bản gốc chỉ hiểu tiếng Anh; MNRL không cần điểm tương đồng gán tay |
| `Llama-3.2-1B` (bản base) | `Qwen2.5-1.5B-Instruct` | Tiếng Việt tốt, không cần xin quyền truy cập |
| LoRA “< 0,6 triệu tham số” | ~4,4 triệu tham số (r=16, q/k/v/o, 28 lớp) | Con số 0,6M không khớp với mô hình 1-1,5B; vẫn < 0,3% tổng tham số |
| 20 epochs cho LLM | mặc định 3 (`--epochs 20` nếu cần) | Dữ liệu sinh từ ~30 bệnh, 20 epochs dễ học vẹt và làm giảm khả năng ngôn ngữ chung |
| Nạp fp16 + Trainer | Nạp fp32 + autocast fp16/bf16, chỉ tính loss trên câu trả lời | Tránh lỗi “unscale FP16 gradients” trên T4 |
| `UnstructuredPDFLoader` | `pypdf` | Không cần poppler/tesseract |
| `ConversationalRetrievalChain` + `ConversationBufferWindowMemory` | Tự viết RAG + bộ nhớ cửa sổ k=3 | API LangChain cũ đã bị loại bỏ; truy xuất lai lọc theo nhãn YOLO chính xác hơn |
| Không có mẫu từ chối | Có mẫu âm dạy LLM từ chối khi tài liệu không liên quan | Giảm ảo giác |

## Mở rộng

- Thêm bệnh mới: thêm mục vào `diseases.json` với `key` = tên thư mục lớp đã chuẩn hóa
  (chữ thường, ký tự đặc biệt → `_`), rồi chạy lại bước 2-4.
- Muốn khoanh **từng vết bệnh**: cần dataset có nhãn vết bệnh (PlantDoc, BRACOL) – đặt trực tiếp vào
  `data/yolo/` theo định dạng YOLO và bỏ qua bước 0.
- Nội dung `diseases.json` là tri thức tham khảo chung; hãy đối chiếu với khuyến cáo của cơ quan bảo vệ thực vật địa phương.
