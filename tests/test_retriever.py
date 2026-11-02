from langchain_core.documents import Document
from plant_doctor.rag.retriever import BM25Index, HybridRetriever, rrf_fuse, tokenize_vi


def D(t, uid=None):
    return Document(page_content=t, metadata={"chunk_uid": uid or t})


def test_tokenize_vi_bigrams():
    assert tokenize_vi("Đạo ôn lúa") == ["đạo", "ôn", "lúa", "đạo_ôn", "ôn_lúa"]


def test_bm25_finds_exact_term():
    idx = BM25Index([D("phun Trichoderma harzianum"), D("tỉa cành thông thoáng")])
    assert idx.search("trichoderma", k=1)[0][0].page_content.startswith("phun")


def test_empty_corpus_returns_empty():
    assert BM25Index([]).search("bất kỳ", k=5) == []


def test_rrf_prefers_docs_in_both_lists():
    a, b, c = D("a"), D("b"), D("c")
    assert rrf_fuse([[a, b], [c, b]])[0].page_content == "b"


def test_gate_drops_low_dense_scores_without_reranker():
    def dense(q, k):
        return [(D("liên quan"), 0.8), (D("lạc đề"), 0.1)]

    r = HybridRetriever(dense=dense, bm25=None, top_k=4)
    assert [d.page_content for d in r.retrieve(["q"], "q")] == ["liên quan"]


def test_reranker_reorders_and_gates():
    class FakeRR:
        def score(self, query, texts):
            return [-2.0 if t == "a" else 3.0 for t in texts]

    def dense(q, k):
        return [(D("a"), 0.9), (D("b"), 0.5)]

    out = HybridRetriever(dense=dense, bm25=None, reranker=FakeRR()).retrieve(["q"], "q")
    assert [d.page_content for d in out] == ["b"]
    assert out[0].metadata["score"] == 3.0
