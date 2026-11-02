"""Bước 5: Ứng dụng Web Streamlit – Giao diện Presentation mỏng gọi qua Service Engine.

streamlit run app/streamlit_app.py
"""
import hashlib
import streamlit as st
import torch
from PIL import Image

from plant_doctor.config import DEFAULT_MAX_CONTEXT_DOCS, DEFAULT_MAX_NEW_TOKENS
from plant_doctor.knowledge import is_healthy
from plant_doctor.pipeline import PlantDoctorPipeline
from plant_doctor.prompts import build_messages, format_context

MEMORY_K = 3         # số lượt hội thoại gần nhất đưa vào prompt (window memory)
MAX_CONTEXT_DOCS = DEFAULT_MAX_CONTEXT_DOCS

st.set_page_config(page_title="Plant Doctor – Chẩn đoán bệnh cây trồng 100% Offline",
                   page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #12301f 0%, #0b1712 45%, #070d0a 100%); }
.hero { padding: 1.6rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, rgba(46,160,96,.25), rgba(16,60,40,.35));
        border: 1px solid rgba(120,220,160,.25); box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.hero h1 { margin: 0; font-weight: 800; font-size: 2rem;
           background: linear-gradient(90deg, #b9f6ca, #69f0ae, #ffe082);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: .4rem 0 0; color: #a5d6b7; }
.badge { display: inline-block; padding: .2rem .7rem; margin: .5rem .4rem 0 0; border-radius: 999px;
         font-size: .78rem; color: #c8f7da; background: rgba(105,240,174,.12);
         border: 1px solid rgba(105,240,174,.3); }
.card { padding: 1rem 1.2rem; border-radius: 16px; background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08); transition: transform .2s, border-color .2s; }
.card:hover { transform: translateY(-2px); border-color: rgba(105,240,174,.45); }
.det { display: flex; justify-content: space-between; padding: .45rem 0;
       border-bottom: 1px dashed rgba(255,255,255,.08); }
.conf { font-weight: 700; color: #69f0ae; }
</style>
<div class="hero">
  <h1>🌿 Plant Doctor – Nhận diện bệnh lá cây & Trợ lý nông nghiệp</h1>
  <p>YOLOv8 tự huấn luyện → Kho tri thức FAISS → Mô hình ngôn ngữ LoRA chạy cục bộ</p>
  <span class="badge">🔒 100% Offline</span><span class="badge">🚫 Không gọi API</span>
  <span class="badge">🧬 Ưu tiên điều trị sinh học</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Nạp Engine (cache)
@st.cache_resource(show_spinner="Đang nạp Plant Doctor Engine (YOLO, FAISS, LLM)...")
def get_pipeline() -> PlantDoctorPipeline:
    return PlantDoctorPipeline(load_models=True)


pipeline = get_pipeline()

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    conf_thr = st.slider("Ngưỡng tin cậy YOLO", 0.10, 0.90, 0.40, 0.05)
    max_new_tokens = st.slider("Độ dài câu trả lời tối đa (token)", 128, 1024, DEFAULT_MAX_NEW_TOKENS, 32)
    st.divider()
    st.subheader("Trạng thái mô hình")
    st.write(("✅" if pipeline.yolo else "❌") + " YOLOv8 (`best.pt`)")
    st.write(("✅" if pipeline.vector_store else "⚠️") + " Embedding + FAISS")
    st.write(("✅" if pipeline.llm else "⚠️") + " LLM LoRA cục bộ")
    if not pipeline.vector_store:
        st.caption("Thiếu Vector DB → dùng thông tin phân tích nhãn.")
    if not pipeline.llm:
        st.caption("Thiếu LLM → trả lời bằng tri thức có cấu trúc (không sinh văn bản).")
    st.caption(f"Thiết bị: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📚 Kho Tri Thức Người Dùng")
    from plant_doctor.rag.document_manager import DocumentManager
    doc_mgr = DocumentManager()
    uploaded_kb = st.file_uploader(
        "Tải lên tài liệu thực tế (.pdf, .md, .txt)",
        type=["pdf", "md", "txt"],
        key="kb_uploader",
    )
    if uploaded_kb is not None:
        if st.button("📥 Nạp vào Vector DB", use_container_width=True):
            with st.spinner("Đang xử lý và phân đoạn tri thức..."):
                doc_mgr.save_uploaded_file(uploaded_kb.name, uploaded_kb.getvalue())
                if pipeline.vector_store and pipeline.embeddings:
                    n_chunks = doc_mgr.ingest_file(uploaded_kb.name, pipeline.vector_store, pipeline.embeddings)
                    st.success(f"Đã nạp {n_chunks} đoạn tri thức từ '{uploaded_kb.name}' vào Vector DB!")
                else:
                    st.success(f"Đã lưu '{uploaded_kb.name}'.")
                st.rerun()

    user_docs = doc_mgr.list_user_documents()
    if user_docs:
        with st.expander(f"Tài liệu đã nạp ({len(user_docs)})"):
            for d in user_docs:
                col_name, col_del = st.columns([4, 1])
                col_name.caption(f"📄 {d['name']} ({d['size_kb']} KB)")
                if col_del.button("❌", key=f"del_{d['name']}", help="Xóa tài liệu này"):
                    doc_mgr.delete_document(d["name"])
                    st.rerun()
    else:
        st.caption("Chưa có tài liệu nào. Hãy tải lên tài liệu PDF/MD thực tế của bạn!")

if pipeline.yolo is None:
    st.error("Không tìm thấy trọng số YOLO. Hãy chạy `python -m plant_doctor.training.train_yolo` trước.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("diagnosis", [])   # [(ten_lop, do_tin_cay, entry|None)]
st.session_state.setdefault("image_key", None)


# ---------------------------------------------------------------- Bộ nhớ hội thoại & Trả lời
def history_pairs() -> list:
    msgs, pairs = st.session_state.messages, []
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs[-MEMORY_K:]


def answer(question: str) -> str:
    """Hiển thị câu trả lời (stream) trong khung chat hiện tại và trả về chuỗi kết quả."""
    diag = st.session_state.diagnosis

    # 1. Tối ưu độ trễ: Kiểm tra Cache phản hồi chuẩn (tức thì 0.01s cho nút Xuất phác đồ)
    cached_text = pipeline.get_cached_answer(question, diag)
    if cached_text:
        st.markdown(cached_text)
        st.caption("⚡ Phản hồi tức thì từ Kho tri thức đã chuẩn hóa (0.01s)")
        return cached_text

    # 2. Truy xuất RAG và suy luận LLM cho câu hỏi tự do
    keys = [e["key"] for _, _, e in diag if e]
    names = ", ".join(c for c, _, _ in diag)
    query = f"{question} {names}".strip()

    docs = pipeline.retrieve_context(query, keys, max_docs=MAX_CONTEXT_DOCS)
    texts = [d if isinstance(d, str) else d.page_content for d in docs]

    if pipeline.llm is None:
        text = pipeline.fallback_answer(diag, texts)
        st.markdown(text)
    else:
        messages = build_messages(format_context(texts), question, history_pairs())
        text = st.write_stream(pipeline.stream_answer(messages, max_new_tokens=max_new_tokens))

    with st.expander(f"📚 Nguồn tri thức đã truy xuất ({len(texts)})"):
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "Tài liệu chuyên môn") if hasattr(d, "metadata") else "Tài liệu tham khảo"
            page = "" if isinstance(d, str) else d.metadata.get("page", "")
            st.markdown(f"**[{i}] {src}** {f'– trang {page}' if page else ''}")
            st.caption((d if isinstance(d, str) else d.page_content)[:600])
    return text


# ---------------------------------------------------------------- Tải ảnh & chẩn đoán
uploaded = st.file_uploader("📸 Tải lên ảnh chiếc lá cần kiểm tra", type=["jpg", "jpeg", "png"],
                            key="leaf_uploader")

if uploaded is not None:
    data = uploaded.getvalue()
    image = Image.open(uploaded).convert("RGB")
    image_key = hashlib.md5(data).hexdigest() + f"_{conf_thr}"

    if st.session_state.image_key != image_key:
        with st.spinner("YOLOv8 đang quét vết bệnh..."):
            diagnosis, annotated = pipeline.diagnose_image(image, conf_thr=conf_thr)
        if st.session_state.image_key and st.session_state.image_key.split("_")[0] != image_key.split("_")[0]:
            st.session_state.messages = []  # ảnh mới -> phiên hội thoại mới
        st.session_state.update(image_key=image_key, diagnosis=diagnosis, annotated=annotated)

    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col1:
        st.markdown("##### Ảnh gốc")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("##### Khoanh vùng bệnh (YOLOv8)")
        if st.session_state.get("annotated") is not None:
            st.image(st.session_state.annotated, use_container_width=True)
    with col3:
        st.markdown("##### Kết quả chẩn đoán")
        diag = st.session_state.diagnosis
        if not diag:
            st.warning("Không phát hiện đối tượng nào vượt ngưỡng tin cậy. Hãy thử ảnh rõ hơn "
                       "hoặc giảm ngưỡng ở thanh bên.")
        else:
            rows = "".join(
                f"<div class='det'><span>{'🟢' if is_healthy(n) else '🔴'} "
                f"{e['name_vi'] + ' – ' + e['crop_vi'] if e else n}</span>"
                f"<span class='conf'>{c:.0%}</span></div>" for n, c, e in diag)
            st.markdown(f"<div class='card'>{rows}</div>", unsafe_allow_html=True)
            if all(is_healthy(n) for n, _, _ in diag):
                st.success("Lá khỏe mạnh – không cần điều trị.")
            unknown = [n for n, _, e in diag if e is None]
            if unknown:
                st.info(f"Chưa có mục tri thức cho: {', '.join(unknown)} – sẽ dùng tìm kiếm ngữ nghĩa.")

    if st.session_state.diagnosis and st.button("📋 Xuất phác đồ điều trị từ Kho tri thức RAG",
                                                type="primary", use_container_width=True):
        names = ", ".join(n for n, _, _ in st.session_state.diagnosis)
        auto_query = (f"Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh: {names}. Hãy giải thích "
                      f"triệu chứng, nguyên nhân gây bệnh và cách điều trị sinh học/phòng ngừa.")
        st.session_state.pending_question = auto_query

# ---------------------------------------------------------------- Hỏi đáp tiếp nối
st.divider()
st.subheader("💬 Hỏi đáp chuyên sâu cùng Trợ lý Nông nghiệp (Offline)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Nhập câu hỏi (VD: Nên phun chế phẩm sinh học vào thời điểm nào?)")
question = st.session_state.pop("pending_question", None) or user_q
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất FAISS và suy luận trên máy cục bộ..."):
            reply = answer(question)
    st.session_state.messages += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": reply}]
