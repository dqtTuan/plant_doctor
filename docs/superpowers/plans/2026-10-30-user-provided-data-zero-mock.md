# Kế Hoạch Triển Khai Hệ Thống Dựa Hoàn Toàn Vào Dữ Liệu Người Dùng (Zero Mock Data)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Loại bỏ hoàn toàn các dữ liệu mẫu/mock data được định sẵn; xây dựng quy trình nạp tài liệu động (Document Ingestion Pipeline) cho phép hệ thống chỉ học, lập chỉ mục vector và trả lời dựa trên **100% tài liệu do chính người dùng thêm vào** (PDF, Word, Markdown, TXT) thông qua giao diện Web hoặc thư mục tri thức, đồng thời kích hoạt cơ chế **Strict Grounding** (từ chối suy diễn nếu tài liệu người dùng nạp không chứa thông tin).

**Architecture:** 
1. Xây dựng module `src/plant_doctor/rag/document_manager.py`: Quản lý vòng đời tài liệu người dùng (lưu trữ, trích xuất văn bản từ PDF/TXT/MD, đếm chunk, lập chỉ mục gia tăng vào FAISS hoặc PostgreSQL `pgvector`, xóa tài liệu).
2. Tích hợp giao diện **"Quản lý & Tải lên Tài liệu Tri thức"** trực tiếp trên Web UI (`app/streamlit_app.py`): Cho phép người dùng kéo thả tài liệu của họ và bấm "Lập chỉ mục ngay".
3. Thiết lập **Strict Grounding** trong `src/plant_doctor/prompts.py` và `src/plant_doctor/pipeline.py`: Loại bỏ toàn bộ các câu trả lời giả lập/mẫu dựng sẵn (mock fallback). Nếu Vector DB rỗng hoặc không tìm thấy tài liệu phù hợp từ dữ liệu người dùng nạp, hệ thống trả lời minh bạch: *"Kho tri thức của bạn chưa có tài liệu về vấn đề này. Vui lòng tải thêm tài liệu chuyên ngành vào hệ thống."*
4. Đảm bảo toàn bộ test suite sử dụng `tmp_path` và file tạm thời độc lập, không phụ thuộc vào bất kỳ file tài liệu mẫu nào trong kho mã nguồn.

**Tech Stack:** Python 3.10+, Streamlit, LangChain Core/Community, PyPDF, FAISS, PostgreSQL + pgvector, Pytest.

**Spec:** Hệ thống vận hành trên 100% dữ liệu người dùng tự thêm, không dùng mock data:
- Kho tài liệu mặc định hoàn toàn sạch sẽ, sẵn sàng nhận file của người dùng.
- Có giao diện và CLI để nạp file của người dùng.
- LLM chỉ trả lời dựa trên những gì có trong tài liệu người dùng nạp (Strict Grounding).

## Global Constraints

- Không hardcode phác đồ hay triệu chứng mẫu trong mã nguồn.
- Tài liệu người dùng nạp (`.pdf`, `.md`, `.txt`) được lưu trữ tại `data/knowledge_base/` và tự động cập nhật vào Vector Store.
- Nếu người dùng chưa nạp tài liệu nào, hệ thống không báo lỗi crash mà hướng dẫn người dùng nạp tài liệu.
- Mọi unit test phải dùng fixture `tmp_path` để tự tạo dữ liệu test giả lập tạm thời, không được dựa vào file tĩnh có sẵn trong repo.
- Đảm bảo 100% test suite `pytest` đều PASS.

## Review Focus

1. **Kho tri thức ban đầu rỗng**: Người dùng mới cài đặt hệ thống chưa nạp file nào $\rightarrow$ ứng dụng không crash, hiển thị thông báo hướng dẫn tải tài liệu.
2. **Người dùng tải file PDF/MD bị lỗi định dạng hoặc rỗng**: Module `document_manager` bắt ngoại lệ, báo lỗi rõ ràng trên UI và không làm hỏng chỉ mục vector hiện có.
3. **Strict Grounding (Không hallucinate)**: Người dùng hỏi một bệnh không có trong các tài liệu họ nạp $\rightarrow$ LLM thông báo rõ ràng không có thông tin trong tài liệu người dùng.
4. **Lập chỉ mục gia tăng (Incremental Indexing)**: Khi người dùng thêm file mới, hệ thống tự động băm nhỏ (Semantic Chunking) và thêm vào Vector Store mà không cần phải xóa toàn bộ dữ liệu cũ.
5. **Hỗ trợ cả FAISS cục bộ lẫn PostgreSQL pgvector**: Hoạt động mượt mà ở cả hai cơ chế lưu trữ khi người dùng nạp tài liệu mới.

---

### Task 1: Module Quản Lý & Nạp Tài Liệu Người Dùng (`document_manager.py`)

**Files:**
- Create: `src/plant_doctor/rag/document_manager.py`
- Test: `tests/test_document_manager.py`

**Interfaces:**
- Produces: Class `DocumentManager`:
  - `save_uploaded_file(file_name: str, file_bytes: bytes) -> Path`: Lưu file người dùng vào `data/knowledge_base/`.
  - `list_user_documents() -> list[dict]`: Liệt kê các tài liệu người dùng đã nạp (tên, dung lượng, thời gian).
  - `ingest_file(file_path: Path, vector_store, embeddings, chunker_type="semantic") -> int`: Phân đoạn và nạp trực tiếp file vào Vector Store.
  - `delete_document(file_name: str) -> bool`: Xóa file tài liệu.

- [x] **Step 1: Viết failing test cho `document_manager.py`**

```python
# tests/test_document_manager.py
import pytest
from pathlib import Path
from plant_doctor.rag.document_manager import DocumentManager

def test_save_uploaded_file(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    saved_path = dm.save_uploaded_file("so_tay_ca_chua.md", b"# Huong dan ca chua")
    assert saved_path.exists()
    assert saved_path.name == "so_tay_ca_chua.md"

def test_list_user_documents(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    dm.save_uploaded_file("doc1.txt", b"Noi dung 1")
    dm.save_uploaded_file("doc2.md", b"Noi dung 2")
    docs = dm.list_user_documents()
    assert len(docs) == 2
    names = [d["name"] for d in docs]
    assert "doc1.txt" in names
    assert "doc2.md" in names

def test_delete_document(tmp_path):
    dm = DocumentManager(kb_dir=tmp_path)
    dm.save_uploaded_file("test.txt", b"xoa toi di")
    assert dm.delete_document("test.txt") is True
    assert len(dm.list_user_documents()) == 0
```

- [x] **Step 2: Chạy test xác nhận fail**

Run: `pytest tests/test_document_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plant_doctor.rag.document_manager'`

- [x] **Step 3: Triển khai `src/plant_doctor/rag/document_manager.py`**

Cài đặt `DocumentManager` với các phương thức lưu file an toàn (tránh path traversal), đọc tài liệu theo phần mở rộng (`.pdf`, `.md`, `.txt`), chia đoạn bằng `chunk_documents_by_type` và cập nhật trực tiếp vào Vector Store.

- [x] **Step 4: Chạy test xác nhận pass**

Run: `pytest tests/test_document_manager.py -v`
Expected: PASS (3/3 tests passed)

- [x] **Step 5: Commit**

```bash
git add src/plant_doctor/rag/document_manager.py tests/test_document_manager.py
git commit -m "feat(rag): add DocumentManager for user-provided document ingestion"
```

---

### Task 2: Thiết Lập Chế Độ Strict Grounding (Từ Chối Trả Lời Khi Thiếu Tài Liệu Người Dùng)

**Files:**
- Modify: `src/plant_doctor/prompts.py`
- Modify: `src/plant_doctor/pipeline.py`
- Test: `tests/test_strict_grounding.py`

**Interfaces:**
- Modifies:
  - `prompts.SYSTEM_PROMPT`: Thắt chặt nguyên tắc Strict Grounding: *Chỉ được sử dụng thông tin có trong tài liệu người dùng cung cấp đính kèm trong prompt. Nếu tài liệu không chứa câu trả lời, dứt khoát thông báo "Tài liệu người dùng cung cấp không có thông tin về vấn đề này". Không được tự bịa ra phác đồ ngoài tài liệu.*
  - `PlantDoctorPipeline.fallback_answer`: Khi không có LLM và không có tài liệu vector từ người dùng, trả về thông báo rõ ràng: *"Chưa có tài liệu về bệnh này trong kho tri thức của bạn. Vui lòng nạp thêm tài liệu."*
  - `PlantDoctorPipeline.get_cached_answer`: Trả về `None` khi không có dữ liệu để chuyển sang RAG từ tài liệu người dùng.

- [x] **Step 1: Viết failing test cho Strict Grounding**

```python
# tests/test_strict_grounding.py
from plant_doctor.pipeline import PlantDoctorPipeline
from plant_doctor.prompts import SYSTEM_PROMPT

def test_system_prompt_enforces_strict_grounding():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "chỉ sử dụng thông tin trong tài liệu" in prompt_lower or "căn cứ vào tài liệu" in prompt_lower
    assert "tuyệt đối không" in prompt_lower or "không tự suy diễn" in prompt_lower or "từ chối" in prompt_lower

def test_fallback_answer_when_no_user_documents():
    pipeline = PlantDoctorPipeline(load_models=False)
    # Giả lập phát hiện bệnh nhưng không có tài liệu nào trong kho
    diag = [("Coffee___Rust", 0.9, {"display_name": "Bệnh gỉ sắt cà phê"})]
    ans = pipeline.fallback_answer(diag, texts=[])
    assert "chưa có tài liệu" in ans.lower() or "vui lòng nạp thêm" in ans.lower()
```

- [x] **Step 2: Chạy test xác nhận fail**

Run: `pytest tests/test_strict_grounding.py -v`
Expected: FAIL

- [x] **Step 3: Cập nhật `src/plant_doctor/prompts.py` và `src/plant_doctor/pipeline.py`**

Điều chỉnh `SYSTEM_PROMPT` nhấn mạnh tính trung thực khoa học nông nghiệp dựa trên dữ liệu người dùng. Cập nhật `fallback_answer` và `retrieve_context` xử lý tình huống kho tài liệu rỗng hoặc chưa khớp.

- [x] **Step 4: Chạy test xác nhận pass**

Run: `pytest tests/test_strict_grounding.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/plant_doctor/prompts.py src/plant_doctor/pipeline.py tests/test_strict_grounding.py
git commit -m "feat(rag): enforce strict grounding on user-provided documents"
```

---

### Task 3: Tích Hợp Giao Diện Quản Lý & Tải Tài Liệu Vào Streamlit App

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_app_import.py`

**Interfaces:**
- Consumes: `DocumentManager`
- Modifies:
  - Thêm tab hoặc khu vực bên Sidebar/Trang chính: **"📂 Quản lý Kho Tri Thức Người Dùng"**:
    - Nút tải lên file (`.pdf`, `.md`, `.txt`).
    - Nút bấm *"Nạp & Cập nhật Vector DB"*: Tự động băm nhỏ và cập nhật ngay vào Vector Store (FAISS/pgvector) trong phiên làm việc.
    - Danh sách tài liệu hiện có kèm dung lượng và nút xóa.
    - Badge hiển thị số lượng tài liệu người dùng đã nạp.

- [x] **Step 1: Viết test kiểm tra Streamlit app import và tương thích UI**
- [x] **Step 2: Cập nhật `app/streamlit_app.py` bổ sung khu vực Ingestion**
- [x] **Step 3: Chạy test xác nhận không có lỗi cú pháp hoặc import**

Run: `pytest tests/test_app_import.py -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add app/streamlit_app.py tests/test_app_import.py
git commit -m "feat(ui): add user document management and upload panel in Streamlit"
```

---

### Task 4: Dọn Dẹp Kho Tri Thức & Độc Lập Hóa Toàn Bộ Test Suite

**Files:**
- Delete hoặc làm sạch: `data/knowledge_base/cam_nang_benh_cay_trong.md` (chỉ giữ lại file hướng dẫn README hoặc để trống cho người dùng nạp)
- Modify: `tests/test_unstructured_rag.py`
- Modify: `tests/test_training_interfaces.py`
- Modify: `src/plant_doctor/rag/build_index.py`

**Interfaces:**
- Đảm bảo các hàm lập chỉ mục và huấn luyện hoạt động an toàn ngay cả khi thư mục `data/knowledge_base/` ban đầu chưa có file nào.
- Toàn bộ unit tests tự động tạo tài liệu tạm thời thông qua fixture `tmp_path`.

- [x] **Step 1: Cập nhật các test suite dùng `tmp_path` tự sinh dữ liệu kiểm thử**
- [x] **Step 2: Chạy kiểm thử toàn bộ `pytest`**
- [x] **Step 3: Commit**

```bash
git add tests/ src/
git commit -m "refactor(test): decouple test suite from static knowledge documents using tmp_path"
```

---

### Task 5: Kiểm Thử Toàn Diện Hệ Thống & Cập Nhật Hướng Dẫn (`README.md`)

**Files:**
- Modify: `README.md`
- Test: Toàn bộ test suite `pytest`

- [x] **Step 1: Chạy toàn bộ test suite dự án**

Run: `pytest`
Expected: 100% tests PASS (30+ tests).

- [x] **Step 2: Cập nhật `README.md`**
  - Hướng dẫn người dùng cách tải tài liệu PDF/MD của chính họ vào hệ thống qua Web UI hoặc CLI.
  - Giải thích cơ chế Strict Grounding: hệ thống chỉ trả lời đúng những gì người dùng cung cấp.

- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document user-provided document ingestion and strict grounding"
```
