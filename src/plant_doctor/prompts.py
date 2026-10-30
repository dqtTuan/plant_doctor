"""Khuôn mẫu prompt cho LLM.

Prompt dùng khi fine-tune LLM và khi suy luận trong app PHẢI giống nhau,
vì vậy cả hai đều gọi `build_messages` trong file này.
"""

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh kiêm chuyên gia bệnh học thực vật tư vấn cho người dân Việt Nam.\n"
    "- Nguyên tắc Strict Grounding: Với câu hỏi về bệnh cây trồng, phác đồ điều trị và nông nghiệp, bạn CHỈ ĐƯỢC PHÉP căn cứ vào tài liệu tham khảo do người dùng cung cấp đính kèm. Tuyệt đối không tự suy diễn hoặc bịa đặt tên thuốc, hoạt chất hay phác đồ ngoài tài liệu. Nếu tài liệu tham khảo ghi '(Không tìm thấy tài liệu phù hợp)' hoặc không có thông tin, bạn phải dứt khoát từ chối và thông báo: 'Tài liệu người dùng cung cấp chưa có thông tin về vấn đề này. Vui lòng nạp thêm tài liệu chuyên ngành.'\n"
    "- Với các câu hỏi chào hỏi, giao tiếp thông thường hoặc kiến thức đời sống, khoa học cơ bản ngoài nông nghiệp: Hãy trả lời tự nhiên, lịch sự, ngắn gọn và hữu ích dựa trên hiểu biết chung."
)


def format_context(texts: list) -> str:
    if not texts:
        return "(Không tìm thấy tài liệu phù hợp)"
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))


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
