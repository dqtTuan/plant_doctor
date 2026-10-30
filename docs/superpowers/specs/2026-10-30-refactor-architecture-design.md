# Thiết Kế Chuẩn Hóa Kiến Trúc 3 Tầng – Plant Doctor

- **Ngày thiết kế**: 2026-10-30
- **Trạng thái**: Đã phê duyệt hướng tiếp cận (Hướng 1: 3-Layered Architecture)
- **Mục tiêu**: Tách bạch hoàn toàn logic suy luận, kho tri thức và pipeline xử lý AI ra khỏi giao diện người dùng Streamlit; chuẩn hóa các module training thành các callable functions; giữ nguyên tính tương thích 100% offline.

---

## 1. Bối cảnh & Vấn đề hiện tại
- **Giao diện béo (Fat UI)**: File `app/streamlit_app.py` hiện chứa ~266 dòng code, ôm đồm: nạp model (`yolo`, `vector_store`, `llm`), thuật toán suy luận YOLO và lọc nhãn, thuật toán truy xuất lai RAG (`retrieve`), tạo luồng streaming generator cho Transformers, quản lý bộ nhớ session, và vẽ giao diện HTML/CSS.
- **Khó mở rộng**: Không thể tái sử dụng logic chẩn đoán/tư vấn này nếu muốn xuất API (FastAPI) hoặc viết CLI chạy batch ảnh.
- **Script training**: Các script `train_yolo.py`, `train_embedding.py`, `train_llm.py` viết thẳng logic vào `main()`, khó kiểm thử đơn vị hoặc gọi từ script khác.

---

## 2. Kiến trúc Đề xuất (3-Layered Architecture)

### Tầng 1: Presentation Layer (Giao diện)
* `app/streamlit_app.py`: Chỉ đóng vai trò nhận diện tương tác của người dùng (tải ảnh, nhập chat, chọn slider) và hiển thị kết quả. Gọi xuống `PlantDoctorPipeline` để thực hiện toàn bộ logic.

### Tầng 2: Service / Engine Layer (Pipeline suy luận)
Tạo module mới `src/plant_doctor/pipeline.py` (hoặc `engine.py`), định nghĩa class trung tâm `PlantDoctorPipeline`:
- `load_models()`: Quản lý nạp an toàn các mô hình (YOLO, Vector Store, LLM) với graceful degradation (nếu thiếu thành phần nào thì tự fallback).
- `diagnose_image(image, conf_thr) -> DiagnosisResult`:
  - Chạy YOLOv8.
  - Ánh xạ nhãn qua `knowledge.lookup()`.
  - Trả về danh sách chẩn đoán có cấu trúc: nhãn, độ tin cậy, thông tin bệnh từ KB, ảnh đã vẽ bounding box.
- `retrieve_context(query, diagnosis_keys, max_docs=4) -> list[Document]`:
  - Thực hiện thuật toán Hybrid Retrieval (Symbolic Filter theo keys của YOLO + Semantic Search).
- `generate_response_stream(question, context_docs, history_pairs) -> Iterator[str]`:
  - Đóng gói prompt qua `build_messages()`.
  - Sinh stream token qua `TextIteratorStreamer`.
- `generate_fallback_response(diagnosis) -> str`:
  - Trường hợp không có LLM cục bộ, sinh phác đồ cấu trúc từ `knowledge.entry_to_answer()`.

### Tầng 3: Core Domain & Training Modules
- `src/plant_doctor/prompts.py`: Lưu System Prompt linh hoạt (vừa hỗ trợ bệnh cây, vừa hỗ trợ giao tiếp cơ bản) và `build_messages()`.
- `src/plant_doctor/knowledge.py`: Tra cứu tri thức, ánh xạ nhãn, chuẩn hóa văn bản.
- `src/plant_doctor/rag/`:
  - `build_index.py`: Tách thành hàm `build_faiss_index(...)` có thể nhận tham số, và `main()` làm CLI wrapper.
- `src/plant_doctor/training/`:
  - Chuẩn hóa các hàm `train_yolo(...)`, `train_embedding(...)`, `train_llm(...)` nhận tham số rõ ràng.

---

## 3. Cấu trúc Thư mục sau Tái cấu trúc

```text
src/plant_doctor/
├── config.py              # Quản lý đường dẫn, hyperparams, environment
├── knowledge.py           # Định dạng tri thức, lookup, alias index
├── prompts.py             # System prompt, format context, build messages
├── pipeline.py            # [MỚI] Engine tổng hợp: YOLO detection, RAG retrieval, LLM streaming
├── data/
│   └── prepare_dataset.py # Bước 0: Tự sinh bbox, chia train/val/test
├── rag/
│   └── build_index.py     # Bước 4: Chunking, embedding, build FAISS
└── training/
    ├── train_yolo.py      # Bước 1: Huấn luyện YOLOv8
    ├── train_embedding.py # Bước 2: Huấn luyện sentence-transformers MNRL
    └── train_llm.py       # Bước 3: Fine-tune Qwen2.5 LoRA & merge

app/
└── streamlit_app.py       # Giao diện mỏng (Thin UI), chỉ render components & gọi pipeline

tests/
├── test_knowledge.py      # Test kho tri thức & prompt
└── test_pipeline.py       # [MỚI] Test pipeline suy luận (mock / unit test)
```

---

## 4. Các tiếp cận đã xem xét (Approaches)
1. **Hướng 1 (Đã chọn)**: Chuẩn hóa 3 tầng (Layered Architecture). Tách Service Engine `pipeline.py`, giữ UI mỏng. Tối ưu cho dự án cục bộ, dễ bảo trì, dễ viết thêm API.
2. **Hướng 2 (Loại bỏ)**: Chia thành nhiều micro-packages riêng biệt (`plant-doctor-vision`, `plant-doctor-rag`, `plant-doctor-llm`). Quá cồng kềnh, gây phân mảnh dependencies không cần thiết.

---

## 5. Kế hoạch Kiểm thử & Đảm bảo Tính tương thích
- Giữ nguyên toàn bộ CLI commands trong `pyproject.toml` (`pd-prepare-dataset`, `pd-train-yolo`,...).
- Chạy `python -m pytest` để đảm bảo 100% tests hiện tại và test mới đều pass.
- Test chạy thử `streamlit run app/streamlit_app.py` để xác nhận giao diện hoạt động mượt mà.
