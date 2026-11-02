# Kế Hoạch Tối Ưu Hóa Tốc Độ Phản Hồi (Inference Latency Optimization Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giảm thiểu độ trễ trả lời (Inference Latency) của hệ thống Plant Doctor bằng cách: tinh giản độ dài ngữ cảnh đưa vào Prompt (Context Pruning), hạ giới hạn sinh token mặc định, bổ sung Exact Response Cache cho các yêu cầu xuất phác đồ định sẵn (tức thì 0s), và tối ưu tham số suy luận LLM.

**Architecture:** Tích hợp tầng cache phản hồi chính xác (`ExactResponseCache`) trong `src/plant_doctor/pipeline.py` nhằm phản hồi tức thì các yêu cầu xuất phác đồ từ YOLO. Thu gọn số lượng tài liệu ngữ cảnh `MAX_CONTEXT_DOCS` từ 4 xuống 2 và điều chỉnh `max_new_tokens` mặc định từ 512 xuống 256. Bật tối ưu hóa nạp mô hình (attention flags / torch evaluation modes).

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, Streamlit, Pytest.

**Spec:** Dựa trên khuyến nghị kỹ thuật tối ưu hóa độ trễ: Context Pruning, Response Caching, Generation Hyperparameter Tuning.

## Global Constraints

- Chạy 100% offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Không làm thay đổi định dạng Markdown đầu ra của các câu trả lời phác đồ chuẩn.
- Toàn bộ các bài kiểm tra `pytest` phải chạy hoàn tất trên CPU trong vòng dưới 30 giây.

## Review Focus

1. **Hiệu lực của Cache**: Khi bấm nút "Xuất phác đồ điều trị từ Kho tri thức RAG", pipeline phải nhận diện được mẫu câu hỏi chuẩn và trả về ngay kết quả đã định dạng mà không kích hoạt quá trình sinh LLM tốn thời gian.
2. **Không chặn câu hỏi tự do**: Cache chỉ áp dụng cho câu hỏi phác đồ chuẩn; các câu hỏi chat tự do hoặc hỏi sâu hơn vẫn phải đi qua luồng RAG + LLM bình thường.
3. **Giảm Context không làm mất thông tin**: Giảm từ 4 xuống 2 tài liệu nhưng 2 tài liệu đó phải là tài liệu liên quan nhất (Symbolic match từ YOLO và Semantic match hàng đầu).
4. **Không vượt giới hạn token**: `max_new_tokens` giảm xuống 256 nhưng câu trả lời không được cụt ý giữa chừng (vẫn kết thúc trọn vẹn).
5. **Tính nhất quán giữa Streamlit UI và Pipeline**: Slider và tham số trong `streamlit_app.py` phải khớp với cấu hình mặc định tối ưu mới.

---

### Task 1: Tối ưu Tham số Ngữ cảnh & Cấu hình Sinh Token (Context Pruning & Generation Tuning)

**Files:**
- Modify: `src/plant_doctor/config.py`
- Modify: `src/plant_doctor/pipeline.py`
- Modify: `app/streamlit_app.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `config.DEFAULT_MAX_NEW_TOKENS`, `config.DEFAULT_MAX_CONTEXT_DOCS`
- Produces: 
  - `pipeline.retrieve_context(..., max_docs=2)`
  - `pipeline.stream_answer(..., max_new_tokens=256)`

- [ ] **Step 1: Viết failing test kiểm tra giá trị mặc định tối ưu mới của pipeline**

Bổ sung test trong `tests/test_pipeline.py`:
```python
def test_pipeline_default_optimization_constants():
    from plant_doctor.config import DEFAULT_MAX_CONTEXT_DOCS, DEFAULT_MAX_NEW_TOKENS
    assert DEFAULT_MAX_CONTEXT_DOCS == 2
    assert DEFAULT_MAX_NEW_TOKENS == 256
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_pipeline.py -k test_pipeline_default_optimization_constants`
Expected: FAIL (AttributeError: module 'plant_doctor.config' has no attribute 'DEFAULT_MAX_CONTEXT_DOCS')

- [ ] **Step 3: Cập nhật `src/plant_doctor/config.py`, `src/plant_doctor/pipeline.py` và `app/streamlit_app.py`**

- Trong `config.py`: Định nghĩa `DEFAULT_MAX_CONTEXT_DOCS = 2` và `DEFAULT_MAX_NEW_TOKENS = 256`.
- Trong `pipeline.py`: Đặt mặc định `retrieve_context(..., max_docs=DEFAULT_MAX_CONTEXT_DOCS)` và `stream_answer(..., max_new_tokens=DEFAULT_MAX_NEW_TOKENS)`.
- Trong `streamlit_app.py`: Đổi slider `max_new_tokens` mặc định thành `256`, `MAX_CONTEXT_DOCS = DEFAULT_MAX_CONTEXT_DOCS`.

- [ ] **Step 4: Chạy test xác nhận pass**

Run: `python -m pytest tests/test_pipeline.py -k test_pipeline_default_optimization_constants`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/config.py src/plant_doctor/pipeline.py app/streamlit_app.py tests/test_pipeline.py
git commit -m "perf: prune default context docs to 2 and default new tokens to 256"
```

---

### Task 2: Xây dựng Cơ chế Response Cache cho Phác Đồ Mặc Định (Exact Response Cache)

**Files:**
- Modify: `src/plant_doctor/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `knowledge.entry_to_answer`, `knowledge.lookup`
- Produces: 
  - `pipeline.get_cached_answer(question: str, diagnosis: list[tuple[str, float, dict | None]]) -> str | None`

- [ ] **Step 1: Viết failing test cho `get_cached_answer`**

Bổ sung test trong `tests/test_pipeline.py`:
```python
def test_get_cached_answer_for_standard_query():
    pipeline = PlantDoctorPipeline(load_models=False)
    fake_entry = {
        "name_vi": "Gỉ sắt", "name_en": "Rust", "crop_vi": "Cà phê", "crop_en": "Coffee",
        "pathogen": "Nấm", "symptoms": "Bột vàng cam", "causes": "Mưa ẩm",
        "bio_treatment": "Phun Bordeaux", "chemical_note": "Hạn chế", "prevention": "Tỉa cành"
    }
    diagnosis = [("Rust", 0.92, fake_entry)]
    auto_query = ("Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh: Rust. Hãy giải thích "
                  "triệu chứng, nguyên nhân gây bệnh và cách điều trị sinh học/phòng ngừa.")
    cached = pipeline.get_cached_answer(auto_query, diagnosis)
    assert cached is not None
    assert "Gỉ sắt" in cached
    assert "Phác đồ sinh học" in cached

def test_get_cached_answer_returns_none_for_free_question():
    pipeline = PlantDoctorPipeline(load_models=False)
    diagnosis = [("Rust", 0.92, None)]
    cached = pipeline.get_cached_answer("Hôm nay trời đẹp không?", diagnosis)
    assert cached is None
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_pipeline.py -k "test_get_cached_answer"`
Expected: FAIL (AttributeError: 'PlantDoctorPipeline' object has no attribute 'get_cached_answer')

- [ ] **Step 3: Triển khai `get_cached_answer` trong `src/plant_doctor/pipeline.py`**

- Kiểm tra nếu `question` bắt đầu bằng cụm từ sinh tự động `"Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh:"`:
  - Trích xuất thông tin bệnh từ danh sách `diagnosis`.
  - Nếu có mục thông tin trong KB, gọi ngay `self.fallback_answer(diagnosis, [])` và trả về kết quả định dạng sẵn (thời gian xử lý < 1ms, không cần chờ LLM sinh token).
- Nếu không phải câu hỏi xuất phác đồ chuẩn, trả về `None` để chuyển cho luồng LLM sinh tự do.

- [ ] **Step 4: Chạy lại test**

Run: `python -m pytest tests/test_pipeline.py -k "test_get_cached_answer"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add exact response cache for standard diagnosis queries"
```

---

### Task 3: Tích hợp Response Cache vào Giao diện Streamlit App

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_app_import.py`

**Interfaces:**
- Consumes: `pipeline.get_cached_answer(question, diag)`
- Produces: Luồng phản hồi trực tiếp khi phát hiện câu hỏi trong cache, bỏ qua spinner chờ mô hình suy luận.

- [ ] **Step 1: Viết test kiểm tra luồng trả lời hỗ trợ cache trong app**

Bổ sung test trong `tests/test_app_import.py`:
```python
def test_streamlit_app_answer_function_exists():
    import app.streamlit_app as st_app
    assert callable(st_app.answer)
```

- [ ] **Step 2: Chạy test**

Run: `python -m pytest tests/test_app_import.py`
Expected: PASS

- [ ] **Step 3: Cập nhật hàm `answer(question)` trong `app/streamlit_app.py`**

- Ngay đầu hàm `answer(question: str)`:
  ```python
  cached_text = pipeline.get_cached_answer(question, diag)
  if cached_text:
      st.markdown(cached_text)
      st.caption("⚡ Trả lời tức thì từ Kho tri thức đã chuẩn hóa (0.01s)")
      return cached_text
  ```
- Nếu không có trong cache, tiếp tục chạy luồng `retrieve_context` và `stream_answer` như bình thường.

- [ ] **Step 4: Chạy toàn bộ test suite**

Run: `python -m pytest`
Expected: 100% tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/test_app_import.py
git commit -m "feat(app): bypass LLM generation on cached standard diagnosis queries"
```

---

### Task 4: Kiểm thử Tích hợp và Đánh giá Hiệu năng

**Files:**
- Modify: `README.md`
- Test: `tests/`

- [ ] **Step 1: Chạy toàn bộ test suite dự án**

Run: `python -m pytest -v`
Expected: PASS toàn bộ các bài kiểm tra.

- [ ] **Step 2: Cập nhật phần hướng dẫn tối ưu độ trễ trong `README.md`**

Bổ sung mục mô tả về Exact Response Caching và các tham số điều chỉnh độ trễ trong `README.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document latency optimization techniques and caching layer"
```
