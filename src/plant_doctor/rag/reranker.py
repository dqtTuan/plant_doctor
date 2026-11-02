"""Module mô hình Cross-Encoder Reranker chạy offline trên máy cục bộ."""
import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        ...


class CrossEncoderReranker:
    def __init__(self, model_dir: Path | str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(model_dir), max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


def load_reranker(model_dir: Path | str) -> Reranker | None:
    """Nạp mô hình Cross-Encoder Reranker nếu file trọng số tồn tại trên ổ cứng."""
    p = Path(model_dir)
    if not (p / "config.json").exists():
        return None
    try:
        return CrossEncoderReranker(p)
    except Exception as e:
        logger.warning("Không thể nạp Cross-Encoder Reranker tại %s: %s", p, e)
        return None
