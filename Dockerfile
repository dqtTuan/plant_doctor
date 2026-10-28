# syntax=docker/dockerfile:1.7
# ------------------------------------------------------------------------------
# Plant Doctor – Streamlit app (100% offline)
#
# Một Dockerfile cho cả 2 biến thể, chọn bằng build-arg TORCH_INDEX_URL:
#   CPU (mặc định): docker build -t plant-doctor .
#   GPU (CUDA)    : docker build -t plant-doctor:gpu \
#                     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
# Wheel CUDA của PyTorch đã kèm sẵn runtime CUDA/cuDNN -> chỉ cần NVIDIA driver +
# NVIDIA Container Toolkit trên máy chủ, không cần image nền nvidia/cuda.
#
# Mô hình KHÔNG nằm trong image: mount models/ và vector_store/ khi chạy
# (xem docker-compose.yml).
# ------------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # config.py đọc biến này để xác định thư mục gốc (package cài vào site-packages)
    PLANT_DOCTOR_ROOT=/app \
    HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgl1 + libglib2.0-0: cần cho opencv-python (phụ thuộc kéo theo của ultralytics)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) PyTorch theo biến thể CPU/GPU (cài trước để pip không kéo bản CUDA mặc định từ PyPI)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

# 2) Thư viện phụ thuộc (layer được cache khi chỉ đổi code)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^-e' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# 3) Package plant_doctor + app + kho tri thức
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-cache-dir . && rm -rf src
COPY app ./app
COPY data/knowledge_base ./data/knowledge_base

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p models vector_store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

