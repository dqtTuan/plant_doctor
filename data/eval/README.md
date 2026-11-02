# Dữ liệu Kiểm Thử Đánh Giá RAG (Golden Retrieval Set)

Thư mục này chứa tập dữ liệu kiểm thử câu hỏi vàng (`golden.jsonl`) dùng để đánh giá độ chính xác của hệ thống truy xuất tri thức (RAG).

## Định dạng file `golden.jsonl`

Mỗi dòng là một đối tượng JSON với cấu trúc:
- `question`: Câu hỏi thực tế của nông dân / người dùng.
- `labels`: Danh sách nhãn phát hiện từ YOLO (tùy chọn, ví dụ `["Tomato___Late_blight"]`).
- `expected`: Danh sách các tài liệu tham khảo đúng chứa câu trả lời (`source`: tên file trong `knowledge_base/`, `page`: số trang nếu là PDF, hoặc `null` nếu là MD/TXT).
  - **Lưu ý:** Nếu `expected: []` (danh sách rỗng), đây là câu hỏi ngoài tài liệu (unanswerable) để kiểm tra cơ chế Strict Grounding từ chối hợp lệ.

### Ví dụ mẫu:
```jsonl
{"question": "Cách phòng trừ bệnh đạo ôn lúa bằng sinh học?", "labels": ["Rice___Blast"], "expected": [{"source": "so_tay_lua.md", "page": null}]}
{"question": "Liều lượng phun Bordeaux cho bệnh gỉ sắt cà phê là bao nhiêu?", "labels": ["Coffee___Rust"], "expected": [{"source": "benh_cay_cong_nghiep.pdf", "page": 12}]}
{"question": "Bệnh lở loét cam có lây sang cây cao su không?", "labels": ["Orange___Citrus_canker"], "expected": []}
```

## Cách chạy đánh giá

```bash
pd-eval-rag --golden data/eval/golden.jsonl --k 4
```
Chỉ số đo lường:
- **Hit@k**: Tỷ lệ câu hỏi tìm thấy tài liệu kỳ vọng trong top-k kết quả.
- **MRR (Mean Reciprocal Rank)**: Điểm xếp hạng tương hỗ của tài liệu đúng đầu tiên.
- **No-Answer Accuracy**: Độ chính xác khi hệ thống từ chối truy xuất đối với các câu hỏi không có tài liệu.
