import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding


class UnitFakeEmbedding(DeterministicFakeEmbedding):
    """Deterministic, L2-normalized (inner product == cosine, like the real model)."""

    def _get_embedding(self, seed=None):
        v = np.asarray(super()._get_embedding(seed=seed))
        return list(v / np.linalg.norm(v))


def test_faiss_inner_product_roundtrip(tmp_path):
    emb = UnitFakeEmbedding(size=16)
    docs = [Document(page_content=t, metadata={"source": "a.md"}) for t in ("gỉ sắt cà phê", "đạo ôn lúa")]
    FAISS.from_documents(docs, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).save_local(str(tmp_path))
    store = FAISS.load_local(
        str(tmp_path),
        emb,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    hit = store.similarity_search("đạo ôn lúa", k=1)[0]
    assert hit.page_content == "đạo ôn lúa"
    assert hit.metadata["source"] == "a.md"
