"""Khuôn mẫu prompt cho LLM.

Prompt dùng khi fine-tune LLM và khi suy luận trong app PHẢI giống nhau,
vì vậy cả hai đều gọi `build_messages` trong file này.
"""

SYSTEM_PROMPT = (
    "Bạn là chuyên gia bệnh học thực vật, tư vấn cho nông dân Việt Nam. "
    "Chỉ được trả lời dựa trên phần 'Tài liệu tham khảo'. "
    "Nếu tài liệu không có thông tin cần thiết, hãy nói rõ là không có và khuyên liên hệ "
    "cán bộ khuyến nông; tuyệt đối không bịa tên thuốc hay liều lượng. "
    "Ưu tiên biện pháp sinh học và canh tác, giảm thiểu thuốc trừ sâu hóa học. "
    "Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc."
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

