# Kế Hoạch Triển Khai Semantic Chunking & Lưu Trữ Vector DB

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa chính xác sơ đồ tuần tự (Sequence Diagram): Đọc tài liệu từ `Document Source` $\rightarrow$ Đưa vào hàm `Generate Chunks Function` bằng thuật toán **Semantic Chunking** (dựa trên khoảng cách ngữ nghĩa giữa các câu liên tiếp) $\rightarrow$ Gửi chunks sang `Generate Embeddings Function` $\rightarrow$ Lưu trữ toàn bộ Vectors kèm Metadata vào `Document DB (FAISS Vector Store)`.

**Architecture:** Bổ sung module `src/plant_doctor/rag/semantic_chunking.py` cài đặt thuật toán Semantic Chunking thuần Python/NumPy (không phụ thuộc thêm thư viện ngoài). Tích hợp vào pipeline đóng gói index trong `src/plant_doctor/rag/build_index.py` cho phép chuyển đổi linh hoạt qua cờ `--chunker-type {semantic, recursive}`.

**Tech Stack:** Python 3.10+, NumPy, PyTorch / Sentence-Transformers, LangChain Core (Document), FAISS.

**Spec:** Sơ đồ tuần tự RAG Chunking: `Document Source` -> `RAG System` -> `Chunk Function (Semantic)` -> `Embedding Function` -> `Document DB (Vector DB with Metadata)`.

## Global Constraints

- Chạy 100% offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Không cài thêm thư viện nặng gây xung đột (tự xây dựng thuật toán Semantic Chunking dựa trên mô hình Embedding cục bộ và NumPy).
- Giữ nguyên cấu trúc lưu trữ của FAISS index (`index.faiss`, `index.pkl`) để `PlantDoctorPipeline` đọc được bình thường.

## Review Focus

1. **Xác định ranh giới ngữ nghĩa (Breakpoint Threshold)**: Ngưỡng chia chunk dựa trên độ sụt giảm khoảng cách Cosine Similarity giữa các câu kế tiếp (phương pháp Percentile hoặc Gradient difference).
2. **Xử lý câu quá ngắn hoặc văn bản đặc biệt**: Tách câu tiếng Việt chuẩn (`.`, `?`, `!`, `\n`) mà không bị ngắt sai tại các từ viết tắt (VD: `TP.HCM`, `pv.`, `spp.`, số thực `1.5`).
3. **Bảo toàn Metadata**: Toàn bộ chunk sau khi chia theo ngữ nghĩa phải giữ nguyên metadata nguồn (`source`, `page`) và được tự động gán `metadata["key"]` qua `tag_disease`.
4. **Giới hạn kích thước chunk (Min/Max Length Guard)**: Tránh sinh ra chunk quá nhỏ (chỉ 1 câu đơn) hoặc quá dài (vượt quá context window của embedding model).
5. **Khả năng tương thích ngược (Backward Compatibility)**: Lệnh `pd-build-index` vẫn hoạt động tốt, hỗ trợ chọn kiểu chunking qua CLI argument.

---

### Task 1: Xây dựng Module Semantic Chunker Thuần Túy (`semantic_chunking.py`)

**Files:**
- Create: `src/plant_doctor/rag/semantic_chunking.py`
- Test: `tests/test_semantic_chunking.py`

**Interfaces:**
- Consumes: `langchain_core.documents.Document`
- Produces: 
  - `split_sentences(text: str) -> list[str]`
  - `semantic_chunking(text: str, embed_fn: Callable[[list[str]], np.ndarray], percentile_threshold: float = 85.0, min_chunk_len: int = 150) -> list[str]`
  - `split_documents_semantically(docs: list[Document], embed_fn: Callable[[list[str]], np.ndarray], percentile_threshold: float = 85.0, kb: list | None = None) -> list[Document]`

- [ ] **Step 1: Viết failing test cho `split_sentences` và `semantic_chunking`**

Tạo `tests/test_semantic_chunking.py`:
```python
import numpy as np
from langchain_core.documents import Document
from plant_doctor.rag.semantic_chunking import split_sentences, semantic_chunking, split_documents_semantically

def test_split_sentences():
    text = "Cây cà phê bị bệnh gỉ sắt. Lá xuất hiện đốm vàng? Cần tỉa cành ngay!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Cây cà phê bị bệnh gỉ sắt."

def test_semantic_chunking_with_mock_embedding():
    # Giả lập 2 nhóm câu khác biệt rõ rệt về vector (nhóm 1: [1, 0], nhóm 2: [0, 1])
    text = "Câu một về cây. Câu hai về cây. Chuyển đề tài về thuốc trừ sâu. Thuốc này rất độc."
    def mock_embed(sentences):
        return np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
        ])
    chunks = semantic_chunking(text, embed_fn=mock_embed, percentile_threshold=50.0, min_chunk_len=10)
    assert len(chunks) == 2
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_semantic_chunking.py`
Expected: FAIL (ModuleNotFoundError: No module named 'plant_doctor.rag.semantic_chunking')

- [ ] **Step 3: Triển khai thuật toán trong `src/plant_doctor/rag/semantic_chunking.py`**

- Hàm `split_sentences(text)`: Sử dụng biểu thức chính quy Regex tách câu thông minh theo dấu chấm, hỏi, than và ngắt dòng kép mà không ngắt nhầm số thập phân hay chữ viết tắt.
- Hàm `semantic_chunking(text, embed_fn, percentile_threshold=85.0, min_chunk_len=150)`:
  1. Tách văn bản thành danh sách câu $S = [s_0, s_1, ..., s_n]$.
  2. Tính vector embedding cho từng câu thông qua `embed_fn`.
  3. Tính khoảng cách Cosine Distance $d_i = 1 - \cos(v_i, v_{i+1})$ giữa các câu liên tiếp.
  4. Xác định ngưỡng cắt (breakpoint threshold) bằng `np.percentile(distances, percentile_threshold)`.
  5. Gom cụm các câu có khoảng cách dưới ngưỡng thành 1 chunk; nếu vượt ngưỡng thì ngắt sang chunk mới (kèm điều kiện bảo vệ `min_chunk_len`).
- Hàm `split_documents_semantically(docs, embed_fn, ...)`: Nhận danh sách Document đầu vào, áp dụng semantic chunking, kế thừa metadata và tự động gắn `tag_disease`.

- [ ] **Step 4: Chạy test xác nhận pass**

Run: `python -m pytest tests/test_semantic_chunking.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/semantic_chunking.py tests/test_semantic_chunking.py
git commit -m "feat(rag): implement semantic chunking algorithm based on sentence embedding distance"
```

---

### Task 2: Tích hợp Semantic Chunking vào Quy trình Đóng gói Vector Database (`build_index.py`)

**Files:**
- Modify: `src/plant_doctor/rag/build_index.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `semantic_chunking.split_documents_semantically`, `config.KB_DIR`, `config.EMBED_DIR`, `config.FAISS_DIR`
- Produces: 
  - `build_faiss_index(..., chunker_type: str = "semantic") -> int`
  - CLI argument `--chunker-type {semantic, recursive}`

- [ ] **Step 1: Viết test cho `build_faiss_index` với chế độ `chunker_type="semantic"`**

Bổ sung test trong `tests/test_rag.py`:
```python
from unittest.mock import MagicMock
from plant_doctor.rag.build_index import chunk_documents_by_type

def test_chunk_documents_by_type_semantic():
    fake_doc = Document(page_content="Cây cà phê bị bệnh gỉ sắt. Lá vàng rụng.", metadata={"source": "doc.txt"})
    mock_embed_fn = lambda sents: np.ones((len(sents), 4))
    chunks = chunk_documents_by_type([fake_doc], chunker_type="semantic", embed_fn=mock_embed_fn)
    assert len(chunks) >= 1
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `python -m pytest tests/test_rag.py -k test_chunk_documents_by_type_semantic`
Expected: FAIL (ImportError / function not found)

- [ ] **Step 3: Cập nhật `src/plant_doctor/rag/build_index.py`**

- Thêm hàm `chunk_documents_by_type(docs, chunker_type="semantic", embed_fn=None, ...)` hỗ trợ chuyển đổi giữa Semantic Chunking và Recursive Splitter.
- Trong `build_faiss_index`: Nạp `HuggingFaceEmbeddings` sớm hơn để tận dụng hàm embedding cho bước Semantic Chunking trước khi đưa toàn bộ chunks vào FAISS Vector Store.
- Cập nhật hàm `main()` với tham số CLI: `--chunker-type` (mặc định `semantic`).

- [ ] **Step 4: Chạy lại test**

Run: `python -m pytest tests/test_rag.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/build_index.py tests/test_rag.py
git commit -m "feat(rag): integrate semantic chunking into FAISS vector database build pipeline"
```

---

### Task 3: Kiểm thử Toàn bộ Hệ thống & Cập nhật Tài liệu Sơ đồ

**Files:**
- Modify: `README.md`
- Test: Toàn bộ test suite `pytest`

- [ ] **Step 1: Chạy toàn bộ test suite dự án**

Run: `python -m pytest -v`
Expected: PASS toàn bộ các bài kiểm tra.

- [ ] **Step 2: Cập nhật tài liệu hướng dẫn trong `README.md`**

Mô tả cơ chế hoạt động của Semantic Chunking, sơ đồ tuần tự lưu trữ vector DB và hướng dẫn lệnh `pd-build-index --chunker-type semantic`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document semantic chunking architecture and vector database integration"
```
