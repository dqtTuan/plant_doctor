"""Bước 5: Ứng dụng Web Streamlit – YOLOv8 -> FAISS + Custom Embedding -> Custom LLM (100% offline).

streamlit run app/streamlit_app.py
"""
import os

# Chặn mọi truy cập HuggingFace Hub: tất cả mô hình phải nằm sẵn trên ổ cứng
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
from threading import Thread

import streamlit as st
import torch
from PIL import Image

from plant_doctor.config import EMBED_DIR, FAISS_DIR, LLM_MERGED_DIR, YOLO_WEIGHTS
from plant_doctor.knowledge import entry_to_answer, entry_to_text, is_healthy, lookup
from plant_doctor.prompts import build_messages, format_context

MEMORY_K = 3         # số lượt hội thoại gần nhất đưa vào prompt (window memory)
MAX_CONTEXT_DOCS = 4

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
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Nạp mô hình (cache)
@st.cache_resource(show_spinner="Đang nạp YOLOv8...")
def load_yolo():
    if not YOLO_WEIGHTS.exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(YOLO_WEIGHTS))


@st.cache_resource(show_spinner="Đang nạp Embedding + FAISS...")
def load_vector_store():
    if not FAISS_DIR.exists() or not EMBED_DIR.exists():
        return None
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=str(EMBED_DIR),
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True,
                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


@st.cache_resource(show_spinner="Đang nạp mô hình ngôn ngữ cục bộ...")
def load_llm():
    if not (LLM_MERGED_DIR / "config.json").exists():
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(str(LLM_MERGED_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(LLM_MERGED_DIR), torch_dtype=torch.float16 if cuda else torch.float32,
        device_map="auto" if cuda else None)
    return tok, model.eval()


yolo = load_yolo()
vector_store = load_vector_store()
tokenizer, llm = load_llm()

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    conf_thr = st.slider("Ngưỡng tin cậy YOLO", 0.10, 0.90, 0.40, 0.05)
    max_new_tokens = st.slider("Độ dài câu trả lời tối đa (token)", 128, 1024, 512, 64)
    st.divider()
    st.subheader("Trạng thái mô hình")
    st.write(("✅" if yolo else "❌") + " YOLOv8 (`best.pt`)")
    st.write(("✅" if vector_store else "⚠️") + " Embedding + FAISS")
    st.write(("✅" if llm else "⚠️") + " LLM LoRA cục bộ")
    if not vector_store:
        st.caption("Thiếu FAISS → dùng trực tiếp diseases.json.")
    if not llm:
        st.caption("Thiếu LLM → trả lời bằng tri thức có cấu trúc (không sinh văn bản).")
    st.caption(f"Thiết bị: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if yolo is None:
    st.error(f"Không tìm thấy trọng số YOLO tại `{YOLO_WEIGHTS}`. "
             "Hãy chạy `python -m plant_doctor.training.train_yolo` trước.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("diagnosis", [])   # [(ten_lop, do_tin_cay, entry|None)]
st.session_state.setdefault("image_key", None)


# ---------------------------------------------------------------- RAG + sinh văn bản
def retrieve(query: str, keys: list) -> list:
    """Truy xuất lai: ưu tiên tài liệu gắn đúng nhãn bệnh (biểu tượng) + tương đồng ngữ nghĩa."""
    if vector_store is None:  # chế độ dự phòng
        return [entry_to_text(e) for e in (lookup(k) for k in keys) if e]
    docs, seen = [], set()
    for key in keys:
        docs += vector_store.similarity_search(query, k=1, filter={"key": key}, fetch_k=100)
    docs += vector_store.similarity_search(query, k=MAX_CONTEXT_DOCS)
    unique = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)
    return unique[:MAX_CONTEXT_DOCS]


def history_pairs() -> list:
    msgs, pairs = st.session_state.messages, []
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return pairs[-MEMORY_K:]


def stream_llm(messages: list):
    from transformers import TextIteratorStreamer

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False,
                  repetition_penalty=1.1,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    Thread(target=llm.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer


def answer(question: str) -> str:
    """Hiển thị câu trả lời (stream) trong khung chat hiện tại và trả về chuỗi kết quả."""
    diag = st.session_state.diagnosis
    keys = [e["key"] for _, _, e in diag if e]
    names = ", ".join(c for c, _, _ in diag)
    query = f"{question} {names}".strip()
    docs = retrieve(query, keys)
    texts = [d if isinstance(d, str) else d.page_content for d in docs]

    if llm is None:
        entries = [e for _, _, e in diag if e]
        text = "\n\n---\n\n".join(entry_to_answer(e) for e in entries) or \
            "Chưa có mô hình ngôn ngữ cục bộ. Tài liệu liên quan:\n\n" + format_context(texts)
        st.markdown(text)
    else:
        messages = build_messages(format_context(texts), question, history_pairs())
        text = st.write_stream(stream_llm(messages))

    with st.expander(f"📚 Nguồn tri thức đã truy xuất ({len(texts)})"):
        for i, d in enumerate(docs, 1):
            src = "diseases.json" if isinstance(d, str) else d.metadata.get("source", "")
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
            res = yolo.predict(image, imgsz=640, conf=conf_thr, verbose=False)[0]
        best = {}
        for b in res.boxes:
            name, conf = yolo.names[int(b.cls[0])], float(b.conf[0])
            best[name] = max(conf, best.get(name, 0.0))
        diagnosis = [(n, c, lookup(n)) for n, c in sorted(best.items(), key=lambda x: -x[1])]
        if st.session_state.image_key and st.session_state.image_key.split("_")[0] != image_key.split("_")[0]:
            st.session_state.messages = []  # ảnh mới -> phiên hội thoại mới
        st.session_state.update(image_key=image_key, diagnosis=diagnosis,
                                annotated=res.plot()[:, :, ::-1])

    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col1:
        st.markdown("##### Ảnh gốc")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("##### Khoanh vùng bệnh (YOLOv8)")
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
