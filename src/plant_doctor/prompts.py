"""Khuôn mẫu prompt cho LLM và chuẩn hóa trích dẫn nguồn tri thức.

Prompt dùng khi fine-tune LLM và khi suy luận trong app PHẢI giống nhau,
vì vậy cả hai đều gọi `build_messages` trong file này.
"""
import re

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh kiêm chuyên gia bệnh học thực vật tư vấn cho người dân Việt Nam.\n"
    "- Nguyên tắc Strict Grounding: Với câu hỏi về bệnh cây trồng, phác đồ điều trị và nông nghiệp, bạn CHỈ ĐƯỢC PHÉP căn cứ vào tài liệu tham khảo do người dùng cung cấp đính kèm. Tuyệt đối không tự suy diễn hoặc bịa đặt tên thuốc, hoạt chất hay phác đồ ngoài tài liệu. Nếu tài liệu tham khảo ghi '(Không tìm thấy tài liệu phù hợp)' hoặc không có thông tin, bạn phải dứt khoát từ chối và thông báo: 'Tài liệu người dùng cung cấp chưa có thông tin về vấn đề này. Vui lòng nạp thêm tài liệu chuyên ngành.'\n"
    "- Trích dẫn: sau mỗi ý lấy từ tài liệu, ghi số thứ tự tài liệu trong ngoặc vuông, ví dụ [1] hoặc [1][2]. Chỉ dùng số có trong danh sách tài liệu.\n"
    "- Với các câu hỏi chào hỏi, giao tiếp thông thường hoặc kiến thức đời sống, khoa học cơ bản ngoài nông nghiệp: Hãy trả lời tự nhiên, lịch sự, ngắn gọn và hữu ích dựa trên hiểu biết chung."
)


def format_context(docs: list) -> str:
    """Định dạng ngữ cảnh kèm số thứ tự, tên nguồn và số trang."""
    if not docs:
        return "(Không tìm thấy tài liệu phù hợp)"
    formatted = []
    for i, d in enumerate(docs, 1):
        if hasattr(d, "page_content"):
            src = d.metadata.get("source", "Tài liệu")
            page = d.metadata.get("page")
            header = f"[{i}] ({src}, trang {page})" if page else f"[{i}] ({src})"
            formatted.append(f"{header}\n{d.page_content}")
        else:
            formatted.append(f"[{i}] {d}")
    return "\n\n".join(formatted)


def extract_citations(answer: str, n_docs: int) -> list[int]:
    """Trích xuất và lọc các chỉ số trích dẫn hợp lệ [1]..[n] từ câu trả lời."""
    raw_matches = re.findall(r"\[(\d+)\]", answer)
    citations = []
    seen = set()
    for m in raw_matches:
        val = int(m)
        if 1 <= val <= n_docs and val not in seen:
            seen.add(val)
            citations.append(val)
    return citations


def build_messages(context: str, question: str, history: list | None = None) -> list:
    """history: danh sách (câu hỏi, câu trả lời) của các lượt trước (cửa sổ k lượt)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history or []:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append(
        {"role": "user", "content": f"Tài liệu tham khảo:\n{context}\n\nCâu hỏi: {question}"}
    )
    return messages
