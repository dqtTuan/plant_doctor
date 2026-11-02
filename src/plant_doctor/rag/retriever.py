"""Module truy xuất lai (Hybrid Retriever): BM25 + Dense Search + RRF + Relevance Gate."""
from collections import defaultdict
import re
from typing import Any, Callable
import unicodedata

from langchain_core.documents import Document

from plant_doctor.config import (
    DEFAULT_MAX_CONTEXT_DOCS,
    DENSE_MIN_SCORE,
    RERANK_MIN_SCORE,
    RETRIEVE_CANDIDATES,
    RRF_K,
)


def tokenize_vi(text: str) -> list[str]:
    """Phân tách từ tiếng Việt thành danh sách unigram và bigram."""
    norm = unicodedata.normalize("NFC", text).lower()
    words = [w for w in re.split(r"[^\w]+", norm) if w.strip()]
    if not words:
        return []

    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
    return tokens


class BM25Index:
    def __init__(self, docs: list[Document]):
        self.docs = list(docs)
        self.corpus = [tokenize_vi(d.page_content) for d in self.docs]
        self._bm25 = None
        if self.docs and any(len(c) > 0 for c in self.corpus):
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.corpus)
            # Đảm bảo IDF luôn dương cho các kho tài liệu nhỏ hoặc từ phổ biến
            for word, idf_val in list(self._bm25.idf.items()):
                if idf_val <= 0:
                    self._bm25.idf[word] = 0.25

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Tìm kiếm từ khóa BM25 và trả về danh sách Document kèm điểm số dương."""
        if self._bm25 is None or not self.docs:
            return []

        tokens = tokenize_vi(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        scored_pairs = [
            (doc, float(score))
            for doc, score in zip(self.docs, scores)
            if score > 0.0
        ]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:k]


def rrf_fuse(rankings: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Kết hợp nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion (RRF)."""
    scores = defaultdict(float)
    doc_map = {}
    best_dense = defaultdict(float)

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            uid = doc.metadata.get("chunk_uid") or doc.page_content
            scores[uid] += 1.0 / (k + rank)
            if uid not in doc_map:
                doc_map[uid] = doc
            if "score" in doc.metadata:
                best_dense[uid] = max(best_dense[uid], doc.metadata["score"])

    sorted_uids = sorted(scores.keys(), key=lambda u: scores[u], reverse=True)
    fused_docs = []
    for uid in sorted_uids:
        doc = doc_map[uid]
        doc.metadata["rrf"] = scores[uid]
        if uid in best_dense:
            doc.metadata["score"] = best_dense[uid]
        fused_docs.append(doc)
    return fused_docs


class HybridRetriever:
    def __init__(
        self,
        dense: Callable[[str, int], list[tuple[Document, float]]],
        bm25: BM25Index | None = None,
        reranker: Any = None,
        candidates: int = RETRIEVE_CANDIDATES,
        top_k: int = DEFAULT_MAX_CONTEXT_DOCS,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k

    def retrieve(self, queries: list[str], rerank_query: str) -> list[Document]:
        """Truy xuất lai qua BM25 và Dense Search, sau đó rerank và lọc relevance gate."""
        rankings = []

        for q in queries:
            if not q.strip():
                continue
            # 1. Dense search
            dense_results = self.dense(q, self.candidates)
            dense_docs = []
            for d, sc in dense_results:
                d.metadata["score"] = float(sc)
                dense_docs.append(d)
            if dense_docs:
                rankings.append(dense_docs)

            # 2. BM25 search
            if self.bm25 is not None:
                bm25_results = self.bm25.search(q, k=self.candidates)
                bm25_docs = [d for d, _ in bm25_results]
                if bm25_docs:
                    rankings.append(bm25_docs)

        if not rankings:
            return []

        # 3. Hợp nhất RRF
        fused = rrf_fuse(rankings, k=RRF_K)[: self.candidates]
        if not fused:
            return []

        # 4. Reranking hoặc Dense Relevance Gate
        if self.reranker is not None:
            texts = [d.page_content for d in fused]
            scores = self.reranker.score(rerank_query, texts)
            ranked_pairs = []
            for doc, sc in zip(fused, scores):
                doc.metadata["score"] = round(float(sc), 4)
                if sc >= RERANK_MIN_SCORE:
                    ranked_pairs.append((doc, sc))
            ranked_pairs.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked_pairs[: self.top_k]]
        else:
            # Lọc theo ngưỡng Dense Cosine Similarity
            passed = [
                d
                for d in fused
                if d.metadata.get("score", 0.0) >= DENSE_MIN_SCORE
            ]
            return passed[: self.top_k]
