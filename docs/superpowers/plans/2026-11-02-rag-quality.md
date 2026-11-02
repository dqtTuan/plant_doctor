# RAG Quality Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retrieval measurable, correct (persisted, deduplicated, deletable, fully embedded chunks), and stronger (header-aware chunks, BM25 + dense fusion, offline cross-encoder rerank, relevance gate) with `[n]` citations in answers.

**Architecture:** A retrieval eval harness lands first so every later task is judged by numbers. Document loading is unified in `rag/loaders.py`; chunking gains Markdown-header breadcrumbs and a token cap equal to the embedding model's `max_seq_length`; `rag/index_store.py` owns FAISS lifecycle with stable chunk IDs. `rag/retriever.py` fuses BM25 and dense rankings with RRF, optionally reranks with a local cross-encoder, and drops low-relevance hits so strict grounding can refuse. Prompts number context blocks with source/page and the UI maps `[n]` back to sources.

**Tech Stack:** LangChain 1.x (core, community FAISS, text-splitters), sentence-transformers 6 (`CrossEncoder`), `rank-bm25`, FAISS, pgvector, Streamlit, pytest.

**Spec:** `C:\Users\tuanas\.gemini\antigravity\brain\78420005-e058-4c75-8470-1ef9c27c76c9\upgrade_audit.md` (findings R1–R7, Q1–Q6; phases P0 eval, P2–P4).
**Depends on:** `docs/superpowers/plans/2026-11-02-dependency-upgrade.md` completed (LangChain 1.x, bounded deps in `pyproject.toml`).

## Global Constraints

- Runtime remains 100% offline; any model not present on disk is skipped gracefully (never crash, never download at runtime).
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, stored at `models/reranker/`; downloaded only by an explicit one-time CLI.
- Hybrid BM25 applies to the FAISS backend only; pgvector gets dense + rerank + gate.
- Embedding model unchanged (`models/custom_embedding`, `max_seq_length` read from `sentence_bert_config.json`, default 128).
- New deps go into `pyproject.toml` with bounds: `rank-bm25>=0.2.2,<0.3`.
- Tests use `tmp_path` and fake embeddings — no real models, no files from `data/knowledge_base/`.
- Vietnamese user-facing strings stay Vietnamese; the strict-grounding refusal sentence in `SYSTEM_PROMPT` is unchanged.
- Test snippets below are compact for reading; expand one-statement-per-line (no `;`) so `ruff check .` stays clean.

## Review Focus

1. Re-uploading a file with the **same name but changed content** → old chunks removed, new ones added (not merged) — Task 4 test `test_reupload_changed_file_replaces_chunks`.
2. **Empty index** (all docs deleted / fresh install) → BM25 is built from zero docs; `retrieve_context` returns `[]` without raising — Task 5 test `test_empty_corpus_returns_empty`.
3. A **single sentence longer than the token cap** (PDF tables, no punctuation) → hard-split, no chunk over the cap — Task 3 test `test_cap_hard_splits_unpunctuated_text`.
4. LLM cites **non-existent indices** (`[7]` with 3 docs) → ignored by `extract_citations` — Task 7 test.
5. Old FAISS index built **without `chunk_uid` metadata** → manager still loads; deletes fall back to matching `metadata["source"]` — Task 4 test `test_delete_source_on_legacy_index`.

---

### Task 1: Retrieval eval harness

**Files:**
- Create: `src/plant_doctor/rag/evaluate.py`
- Create: `data/eval/README.md` (golden-file format + example lines)
- Modify: `pyproject.toml` (`[project.scripts] pd-eval-rag = "plant_doctor.rag.evaluate:main"`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `GoldenItem` dataclass: `question: str`, `labels: list[str]`, `expected: list[tuple[str, int | None]]` (source filename, page or None). Empty `expected` = unanswerable.
  - `load_golden(path: Path) -> list[GoldenItem]` — JSONL lines `{"question": ..., "labels": [...], "expected": [{"source": "x.pdf", "page": 3}]}`.
  - `evaluate(retrieve: Callable[[str, list[str]], list[Document]], items: list[GoldenItem], k: int = 4) -> dict` with keys `hit_at_k`, `mrr`, `no_answer_acc`, `n_answerable`, `n_unanswerable`.
  - A retrieved doc matches `(source, page)` when `metadata["source"] == source` and (`page is None` or `metadata.get("page") == page`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
from langchain_core.documents import Document
from plant_doctor.rag.evaluate import GoldenItem, evaluate, load_golden

def D(src, page=None):
    return Document(page_content="x", metadata={"source": src, **({"page": page} if page else {})})

def test_metrics_hit_mrr_and_no_answer():
    items = [GoldenItem("q1", [], [("a.pdf", 2)]),       # hit at rank 2
             GoldenItem("q2", [], [("b.md", None)]),     # miss
             GoldenItem("q3", [], [])]                   # unanswerable, retriever returns nothing
    table = {"q1": [D("z.md"), D("a.pdf", 2)], "q2": [D("c.md")], "q3": []}
    m = evaluate(lambda q, labels: table[q], items, k=4)
    assert m["hit_at_k"] == 0.5
    assert m["mrr"] == 0.25
    assert m["no_answer_acc"] == 1.0
    assert (m["n_answerable"], m["n_unanswerable"]) == (2, 1)

def test_load_golden(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"question": "q", "labels": ["Tomato___Late_blight"], '
                 '"expected": [{"source": "a.pdf", "page": 3}]}\n', encoding="utf-8")
    assert load_golden(p) == [GoldenItem("q", ["Tomato___Late_blight"], [("a.pdf", 3)])]
```

- [ ] **Step 2:** `pytest tests/test_evaluate.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement. `main()` args: `--golden` (default `data/eval/golden.jsonl`), `--k 4`, `--json PATH` (optional dump). It builds `PlantDoctorPipeline()` and calls `evaluate(lambda q, l: p.retrieve_context(q, l, max_docs=k), ...)`, printing a 4-line table.
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5:** Write `data/eval/README.md` (format + 2 example lines) and un-ignore `data/eval/` in `.gitignore`. Note: the user adds their own KB documents and 15–30 golden questions (≥3 unanswerable).
- [ ] **Step 6: Record baseline.** If `data/eval/golden.jsonl` exists, run `pd-eval-rag --json data/eval/baseline.json`; otherwise note "baseline pending golden set" in the commit body.
- [ ] **Step 7:** Commit: `feat(eval): retrieval eval harness (hit@k, MRR, no-answer accuracy)`.

---

### Task 2: Unified loaders + stable IDs

**Files:**
- Create: `src/plant_doctor/rag/loaders.py`
- Modify: `src/plant_doctor/rag/build_index.py` (`load_documents` delegates), `src/plant_doctor/rag/document_manager.py` (`load_file_documents` delegates)
- Test: `tests/test_loaders.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_SUFFIXES = (".pdf", ".md", ".txt")`
  - `file_doc_id(path: Path) -> str` — `sha1(file bytes).hexdigest()[:16]`.
  - `load_file(path: Path) -> list[Document]` — metadata `source` (filename), `doc_id`, `page` (PDF only, 1-based). Empty/whitespace pages skipped. Unsupported suffix → `[]`.
  - `load_directory(kb_dir: Path) -> list[Document]` — sorted `rglob`, skips names starting with `readme` (case-insensitive).
  - `assign_chunk_uids(chunks: list[Document]) -> list[Document]` — sets `metadata["chunk_uid"] = sha1(f"{doc_id}:{page or 0}:{i}")[:16]` where `doc_id = metadata.get("doc_id") or metadata.get("source", "")` and `i` is the running index within each `(doc_id, page)` group; mutates and returns.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loaders.py
from langchain_core.documents import Document
from plant_doctor.rag.loaders import assign_chunk_uids, load_directory, load_file

def test_load_file_md_has_source_and_doc_id(tmp_path):
    p = tmp_path / "a.md"; p.write_text("# Lúa\nĐạo ôn", encoding="utf-8")
    [d] = load_file(p)
    assert d.metadata["source"] == "a.md" and len(d.metadata["doc_id"]) == 16

def test_doc_id_changes_with_content(tmp_path):
    p = tmp_path / "a.md"; p.write_text("v1", encoding="utf-8"); id1 = load_file(p)[0].metadata["doc_id"]
    p.write_text("v2", encoding="utf-8"); assert load_file(p)[0].metadata["doc_id"] != id1

def test_load_directory_skips_readme_and_unsupported(tmp_path):
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("y", encoding="utf-8")
    (tmp_path / "b.docx").write_bytes(b"z")
    assert [d.metadata["source"] for d in load_directory(tmp_path)] == ["a.txt"]

def test_chunk_uids_are_stable_and_unique():
    mk = lambda: [Document(page_content=t, metadata={"doc_id": "d1", "page": 1}) for t in ("a", "b")]
    a, b = assign_chunk_uids(mk()), assign_chunk_uids(mk())
    assert [c.metadata["chunk_uid"] for c in a] == [c.metadata["chunk_uid"] for c in b]
    assert len({c.metadata["chunk_uid"] for c in a}) == 2
```

- [ ] **Step 2:** `pytest tests/test_loaders.py -v` → FAIL.
- [ ] **Step 3:** Implement; replace both duplicated loader bodies with calls to `load_directory` / `load_file`.
- [ ] **Step 4:** `pytest -q` → all PASS (existing `test_unstructured_rag`, `test_document_manager` unchanged).
- [ ] **Step 5:** Commit: `refactor(rag): unified loaders with content-hash doc_id and stable chunk_uid`.

---

### Task 3: Header-aware, token-capped chunking

**Files:**
- Modify: `src/plant_doctor/rag/semantic_chunking.py`
- Modify: `src/plant_doctor/rag/build_index.py` (`chunk_documents_by_type`, `build_vector_store`)
- Modify: `src/plant_doctor/config.py`
- Test: `tests/test_chunk_quality.py`

**Interfaces:**
- Consumes: `assign_chunk_uids` (Task 2).
- Produces:
  - `split_markdown_sections(doc: Document) -> list[Document]` — for `source` ending in `.md`, splits on `#`, `##`, `###` (LangChain `MarkdownHeaderTextSplitter`, `strip_headers=True`) and sets `metadata["breadcrumb"] = " › ".join(non-empty h1, h2, h3)`. Other docs are returned unchanged as `[doc]`.
  - `cap_chunks(chunks: list[Document], max_tokens: int, count_tokens: Callable[[str], int]) -> list[Document]` — any chunk whose **embedded text** (breadcrumb line + content) exceeds `max_tokens` is split, first at sentence boundaries (`split_sentences`) and then by hard word windows, until every piece fits. Pieces inherit metadata.
  - `embedded_text(doc: Document) -> str` — `f"{breadcrumb}\n{page_content}"` if a breadcrumb exists, else `page_content`. Used both for embedding and for display.
  - `chunk_documents_by_type(docs, chunker_type="semantic", chunk_size=1000, chunk_overlap=200, embed_fn=None, kb=None, max_tokens: int | None = None, count_tokens: Callable[[str], int] | None = None) -> list[Document]` — order: markdown sections → semantic/recursive → `cap_chunks` (when both cap args are given) → write `embedded_text` into `page_content` → `assign_chunk_uids`.
  - In `config.py`: `embed_max_tokens(embed_dir: Path = EMBED_DIR) -> int` returns `max_seq_length` from `sentence_bert_config.json` minus 2 (CLS/SEP), default `126`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunk_quality.py
from langchain_core.documents import Document
from plant_doctor.rag.semantic_chunking import cap_chunks, embedded_text, split_markdown_sections
from plant_doctor.rag.build_index import chunk_documents_by_type

words = lambda s: len(s.split())

def test_markdown_breadcrumb():
    doc = Document(page_content="# Cà chua\n## Mốc sương\n### Triệu chứng\nVết úng nước.",
                   metadata={"source": "a.md"})
    [s] = split_markdown_sections(doc)
    assert s.metadata["breadcrumb"] == "Cà chua › Mốc sương › Triệu chứng"
    assert embedded_text(s).startswith("Cà chua › Mốc sương › Triệu chứng\n")

def test_cap_splits_long_chunk_on_sentences():
    doc = Document(page_content=" ".join(["Câu số một dài."] * 50), metadata={"source": "a.txt"})
    out = cap_chunks([doc], max_tokens=20, count_tokens=words)
    assert len(out) > 1 and all(words(embedded_text(c)) <= 20 for c in out)

def test_cap_hard_splits_unpunctuated_text():
    doc = Document(page_content="lua " * 300, metadata={"source": "t.pdf", "page": 2})
    out = cap_chunks([doc], max_tokens=50, count_tokens=words)
    assert all(words(c.page_content) <= 50 for c in out) and out[0].metadata["page"] == 2

def test_pipeline_chunks_carry_uid_and_breadcrumb_text():
    doc = Document(page_content="# Lúa\n## Đạo ôn\n" + "Vết bệnh hình thoi. " * 40,
                   metadata={"source": "a.md", "doc_id": "d1"})
    chunks = chunk_documents_by_type([doc], chunker_type="recursive", max_tokens=40, count_tokens=words)
    assert all(c.page_content.startswith("Lúa › Đạo ôn\n") for c in chunks)
    assert all(words(c.page_content) <= 40 for c in chunks)
    assert len({c.metadata["chunk_uid"] for c in chunks}) == len(chunks)
```

- [ ] **Step 2:** `pytest tests/test_chunk_quality.py -v` → FAIL.
- [ ] **Step 3:** Implement. In `build_vector_store` and `DocumentManager.ingest_file`, pass `max_tokens=embed_max_tokens(embed_dir)` and `count_tokens=lambda s: len(tok.encode(s, add_special_tokens=False))`, with `tok = AutoTokenizer.from_pretrained(str(embed_dir))`.
- [ ] **Step 4:** `pytest -q` → all PASS.
- [ ] **Step 5:** Commit: `feat(rag): markdown breadcrumbs and token cap matching embedding max_seq_length`.

---

### Task 4: FAISS lifecycle — persist, dedupe, delete (+ pgvector parity)

**Files:**
- Create: `src/plant_doctor/rag/index_store.py`
- Modify: `src/plant_doctor/rag/document_manager.py` (`ingest_file`, `delete_document`)
- Modify: `src/plant_doctor/rag/pgvector_store.py` (unique `chunk_uid`, `delete_by_source`, HNSW index)
- Modify: `src/plant_doctor/pipeline.py` (`load_vector_store` logs instead of `pass`; holds `self.index_manager`)
- Modify: `app/streamlit_app.py` (upload/delete call the manager)
- Test: `tests/test_index_store.py`

**Interfaces:**
- Consumes: `chunk_uid`, `doc_id`, `source` metadata (Tasks 2–3).
- Produces `FaissIndexManager`:
  - `__init__(index_dir: Path, embeddings: Embeddings)` — loads the index if `index.faiss` exists, otherwise starts empty (store created on first `add`).
  - `store -> FAISS | None` (property)
  - `add(chunks: list[Document]) -> int` — uses `chunk_uid` as FAISS id; skips ids already in the docstore; saves; returns the number added.
  - `delete_source(source: str) -> int` — deletes every id whose `metadata["source"] == source` (works on legacy docs without `chunk_uid`, because ids come from `index_to_docstore_id`); saves; returns the count.
  - `all_documents() -> list[Document]`
  - `version -> int` — increments on every add/delete (consumers rebuild BM25 when it changes).
- `DocumentManager.ingest_file(file_name, index, chunker_type="semantic", max_tokens=None, count_tokens=None) -> int` — `index` is a `FaissIndexManager` or `PGVectorStore` (both expose `.embeddings` for the semantic chunker's `embed_fn`); it calls `delete_source(file_name)` first, then `add`.
- `DocumentManager.delete_document(file_name, index=None) -> bool` — also removes vectors when `index` is given.
- `PGVectorStore`: new optional ctor arg `embeddings: Embeddings | None = None` (stored as `.embeddings`; `add(chunks)` embeds via it), column `chunk_uid TEXT UNIQUE`, `INSERT ... ON CONFLICT (chunk_uid) DO NOTHING`, `delete_by_source(source: str) -> int`, `CREATE INDEX IF NOT EXISTS ... USING hnsw (embedding vector_cosine_ops)`. `PGVectorStore.delete_source = delete_by_source` (alias for interface parity).

- [ ] **Step 1: Write the failing tests** (reuse the `UnitFakeEmbedding` from `tests/test_langchain_compat.py`; move it to `tests/conftest.py` as fixture `fake_emb`)

```python
# tests/test_index_store.py
from langchain_core.documents import Document
from plant_doctor.rag.index_store import FaissIndexManager
from plant_doctor.rag.document_manager import DocumentManager

def chunk(t, src="a.md", uid=None):
    return Document(page_content=t, metadata={"source": src, "chunk_uid": uid or t})

def test_add_persists_across_reload(tmp_path, fake_emb):
    FaissIndexManager(tmp_path, fake_emb).add([chunk("x"), chunk("y")])
    assert len(FaissIndexManager(tmp_path, fake_emb).all_documents()) == 2

def test_add_is_idempotent(tmp_path, fake_emb):
    m = FaissIndexManager(tmp_path, fake_emb)
    assert m.add([chunk("x")]) == 1 and m.add([chunk("x")]) == 0

def test_delete_source_removes_only_that_file(tmp_path, fake_emb):
    m = FaissIndexManager(tmp_path, fake_emb)
    m.add([chunk("x", "a.md"), chunk("y", "b.md")])
    assert m.delete_source("a.md") == 1
    assert [d.metadata["source"] for d in FaissIndexManager(tmp_path, fake_emb).all_documents()] == ["b.md"]

def test_delete_source_on_legacy_index(tmp_path, fake_emb):
    from langchain_community.vectorstores import FAISS
    FAISS.from_documents([Document(page_content="old", metadata={"source": "a.md"})], fake_emb).save_local(str(tmp_path))
    assert FaissIndexManager(tmp_path, fake_emb).delete_source("a.md") == 1

def test_reupload_changed_file_replaces_chunks(tmp_path, fake_emb):
    kb, idx = tmp_path / "kb", FaissIndexManager(tmp_path / "idx", fake_emb)
    dm = DocumentManager(kb_dir=kb)
    dm.save_uploaded_file("a.txt", "Phiên bản một.".encode()); dm.ingest_file("a.txt", idx, chunker_type="recursive")
    dm.save_uploaded_file("a.txt", "Phiên bản hai.".encode()); dm.ingest_file("a.txt", idx, chunker_type="recursive")
    texts = [d.page_content for d in idx.all_documents()]
    assert texts == ["Phiên bản hai."]

def test_delete_document_removes_vectors(tmp_path, fake_emb):
    idx = FaissIndexManager(tmp_path / "idx", fake_emb)
    dm = DocumentManager(kb_dir=tmp_path / "kb")
    dm.save_uploaded_file("a.txt", "Nội dung.".encode()); dm.ingest_file("a.txt", idx, chunker_type="recursive")
    assert dm.delete_document("a.txt", index=idx) is True and idx.all_documents() == []
```

- [ ] **Step 2:** `pytest tests/test_index_store.py -v` → FAIL.
- [ ] **Step 3:** Implement the manager (FAISS `add_documents(docs, ids=[...])`, `delete(ids)`, `save_local`), update `DocumentManager`, and update `pgvector_store.py` (extend `tests/test_pgvector_store.py` with mocked-cursor tests asserting the `ON CONFLICT` and `hnsw` SQL fragments). In `pipeline.load_vector_store`: `logger.warning("pgvector unavailable, falling back to FAISS: %s", e)`; create `self.index_manager = FaissIndexManager(FAISS_DIR, self.embeddings)` and set `self.vector_store = self.index_manager.store`.
- [ ] **Step 4:** In the app, upload calls `doc_mgr.ingest_file(name, pipeline.index_manager or pipeline.pg_store, ...)` and delete calls `doc_mgr.delete_document(name, index=...)`.
- [ ] **Step 5:** `pytest -q` → all PASS.
- [ ] **Step 6:** Commit: `fix(rag): persist FAISS on ingest, dedupe by chunk_uid, delete vectors with file; pgvector upsert + HNSW`.

---

### Task 5: Hybrid retriever (BM25 + dense + RRF + relevance gate)

**Files:**
- Create: `src/plant_doctor/rag/retriever.py`
- Modify: `src/plant_doctor/config.py`, `src/plant_doctor/pipeline.py` (`retrieve_context`), `app/streamlit_app.py` (`answer`), `pyproject.toml` (`rank-bm25>=0.2.2,<0.3`)
- Modify: `tests/test_pipeline.py` (`DEFAULT_MAX_CONTEXT_DOCS` assertion 2 → 4; `test_retrieve_context_open_semantic_search` mocks `similarity_search_with_score` returning `[(Document("Tài liệu chuyên sâu về nấm bệnh từ file cẩm nang Markdown."), 0.9)]` instead of `similarity_search`)
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `FaissIndexManager.all_documents()`, `.version`, `.store` (Task 4).
- Produces:
  - Config: `DEFAULT_MAX_CONTEXT_DOCS = 4`, `RETRIEVE_CANDIDATES = 20`, `RRF_K = 60`, `DENSE_MIN_SCORE = 0.35`, `RERANK_MIN_SCORE = 0.0` (calibrate both with `pd-eval-rag`).
  - `tokenize_vi(text: str) -> list[str]` — NFC, lowercase, split on `[^\w]+`, drop empty strings; returns unigrams plus adjacent bigrams joined with `_` (e.g. `"đạo ôn lúa"` → `["đạo","ôn","lúa","đạo_ôn","ôn_lúa"]`).
  - `BM25Index(docs: list[Document])` with `search(query: str, k: int) -> list[tuple[Document, float]]` (`rank_bm25.BM25Okapi`; returns only score > 0; an empty corpus returns `[]`).
  - `rrf_fuse(rankings: list[list[Document]], k: int = RRF_K) -> list[Document]` — keyed by `metadata.get("chunk_uid") or page_content`; score `Σ 1/(k + rank)` with rank starting at 1; sorted descending; the fused score is stored in `metadata["rrf"]`.
  - `HybridRetriever(dense: Callable[[str, int], list[tuple[Document, float]]], bm25: BM25Index | None, reranker: "Reranker | None" = None, candidates: int = RETRIEVE_CANDIDATES, top_k: int = DEFAULT_MAX_CONTEXT_DOCS)`, with `retrieve(queries: list[str], rerank_query: str) -> list[Document]`:
    1. For each query: get dense top `candidates` and BM25 top `candidates`. Feed all lists into `rrf_fuse`, keep the top `candidates`.
    2. If `reranker`: score `(rerank_query, page_content)`, sort by score, drop items with score `< RERANK_MIN_SCORE`, set `metadata["score"]`.
       Otherwise: drop items whose best dense cosine is `< DENSE_MIN_SCORE` (BM25-only hits are dropped too), and set `metadata["score"]` to that cosine.
    3. Return the first `top_k`.
  - `PlantDoctorPipeline.retrieve_context(query, keys=None, max_docs=DEFAULT_MAX_CONTEXT_DOCS) -> list` — `queries = [query] + [label_to_query(k) for k in keys]`, and `rerank_query = query if not keys else f"{query}. {label_to_query(keys[0])}"`. The BM25 index is used only when `self.index_manager` is set (rebuilt lazily when its `version` changes); a bare `vector_store` (tests, pgvector) runs dense-only. The no-store fallback is unchanged.
  - App `answer()`: pass `question` alone as `query` (stop appending raw class names).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retriever.py
from langchain_core.documents import Document
from plant_doctor.rag.retriever import BM25Index, HybridRetriever, rrf_fuse, tokenize_vi

def D(t, uid=None): return Document(page_content=t, metadata={"chunk_uid": uid or t})

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
    dense = lambda q, k: [(D("liên quan"), 0.8), (D("lạc đề"), 0.1)]
    r = HybridRetriever(dense=dense, bm25=None, top_k=4)
    assert [d.page_content for d in r.retrieve(["q"], "q")] == ["liên quan"]

def test_reranker_reorders_and_gates():
    class FakeRR:
        def score(self, query, texts): return [-2.0 if t == "a" else 3.0 for t in texts]
    dense = lambda q, k: [(D("a"), 0.9), (D("b"), 0.5)]
    out = HybridRetriever(dense=dense, bm25=None, reranker=FakeRR()).retrieve(["q"], "q")
    assert [d.page_content for d in out] == ["b"] and out[0].metadata["score"] == 3.0
```

- [ ] **Step 2:** `pytest tests/test_retriever.py -v` → FAIL.
- [ ] **Step 3:** Implement the retriever and wire it into the pipeline. The FAISS dense callable is `store.similarity_search_with_score(q, k)`; with `MAX_INNER_PRODUCT` the score is already the cosine. The pgvector callable returns `1 - distance` (add `similarity_search_with_score(query_vector, k)` to `PGVectorStore`).
- [ ] **Step 4:** Update `tests/test_pipeline.py::test_pipeline_default_optimization_constants` to `== 4`. `pytest -q` → all PASS.
- [ ] **Step 5:** If a golden set exists, run `pd-eval-rag --json data/eval/hybrid.json` and put both tables in the commit body.
- [ ] **Step 6:** Commit: `feat(rag): hybrid BM25+dense retrieval with RRF and relevance gate`.

---

### Task 6: Offline cross-encoder reranker

**Files:**
- Create: `src/plant_doctor/rag/reranker.py`, `src/plant_doctor/rag/download_models.py`
- Modify: `src/plant_doctor/config.py`, `src/plant_doctor/pipeline.py` (`load_reranker`, called from `load_all_models`), `app/streamlit_app.py` (sidebar status line), `pyproject.toml` (script `pd-download-reranker = "plant_doctor.rag.download_models:main"`)
- Test: `tests/test_reranker.py`

**Interfaces:**
- Produces:
  - Config: `RERANKER_BASE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"`, `RERANKER_DIR = MODELS_DIR / "reranker"`.
  - `Reranker` Protocol: `score(query: str, texts: list[str]) -> list[float]`.
  - `CrossEncoderReranker(model_dir: Path)` wraps `sentence_transformers.CrossEncoder(str(model_dir), max_length=512)`; `score` returns raw logits as floats.
  - `load_reranker(model_dir: Path = RERANKER_DIR) -> Reranker | None` returns `None` when `model_dir / "config.json"` is missing, and also logs and returns `None` on load errors.
  - `download_models.main()` runs `CrossEncoder(RERANKER_BASE).save(str(RERANKER_DIR))`, temporarily setting `HF_HUB_OFFLINE=0` for this CLI only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reranker.py
from plant_doctor.rag.reranker import load_reranker

def test_missing_model_returns_none(tmp_path):
    assert load_reranker(tmp_path) is None

def test_pipeline_works_without_reranker(monkeypatch, tmp_path):
    import plant_doctor.pipeline as pl
    monkeypatch.setattr(pl, "RERANKER_DIR", tmp_path)
    p = pl.PlantDoctorPipeline(load_models=False)
    assert p.load_reranker() is None
```

- [ ] **Step 2:** `pytest tests/test_reranker.py -v` → FAIL.
- [ ] **Step 3:** Implement. Have `retrieve_context` pass `self.reranker` into `HybridRetriever`.
- [ ] **Step 4:** `pytest -q` → PASS.
- [ ] **Step 5 (manual, needs network once):** `pd-download-reranker` → `models/reranker/config.json` exists. If a golden set exists, run `pd-eval-rag --json data/eval/rerank.json`, then set `RERANK_MIN_SCORE` to the value that maximizes `no_answer_acc` without lowering `hit_at_k` by more than 0.05. Record the chosen value in a config comment.
- [ ] **Step 6:** Commit: `feat(rag): optional offline cross-encoder reranker`.

---

### Task 7: Citations

**Files:**
- Modify: `src/plant_doctor/prompts.py`, `src/plant_doctor/pipeline.py` (`fallback_answer`), `app/streamlit_app.py` (`answer`), `src/plant_doctor/training/train_llm.py` (use the new `format_context` so training matches inference)
- Test: `tests/test_citations.py`

**Interfaces:**
- Produces:
  - `format_context(docs: list) -> str` — for a `Document`: `[n] (source, trang p)\n<page_content>`, or `[n] (source)` when there is no page. For a `str`: `[n] <text>`. An empty list still returns `"(Không tìm thấy tài liệu phù hợp)"`.
  - `SYSTEM_PROMPT` gets one added rule (the existing text is unchanged): `"- Trích dẫn: sau mỗi ý lấy từ tài liệu, ghi số thứ tự tài liệu trong ngoặc vuông, ví dụ [1] hoặc [1][2]. Chỉ dùng số có trong danh sách tài liệu."`
  - `extract_citations(answer: str, n_docs: int) -> list[int]` — finds all `[k]` (including inside `[1][2]`), keeps `1 <= k <= n_docs`, and returns them unique in order of first appearance.
- App: after the answer, the sources expander lists cited docs first labelled `✅ được trích dẫn`, then the rest labelled `· không trích dẫn`, each showing source, page and `metadata["score"]` (2 decimals) when present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_citations.py
from langchain_core.documents import Document
from plant_doctor.prompts import SYSTEM_PROMPT, extract_citations, format_context

def test_format_context_includes_source_and_page():
    ctx = format_context([Document(page_content="Phun Trichoderma.", metadata={"source": "a.pdf", "page": 3}),
                          Document(page_content="Tỉa cành.", metadata={"source": "b.md"})])
    assert ctx.startswith("[1] (a.pdf, trang 3)\nPhun Trichoderma.")
    assert "[2] (b.md)\nTỉa cành." in ctx

def test_format_context_empty_unchanged():
    assert format_context([]) == "(Không tìm thấy tài liệu phù hợp)"

def test_extract_citations_filters_and_dedupes():
    assert extract_citations("Phun sớm [2][1]. Lặp [2]. Sai [7].", n_docs=3) == [2, 1]

def test_prompt_requires_citations():
    assert "[1]" in SYSTEM_PROMPT and "Trích dẫn" in SYSTEM_PROMPT
```

- [ ] **Step 2:** `pytest tests/test_citations.py -v` → FAIL.
- [ ] **Step 3:** Implement. The app passes `docs` (not `texts`) to `format_context` and `fallback_answer`. `fallback_answer(diagnosis, docs)` accepts Documents or strings.
- [ ] **Step 4:** `pytest -q` → all PASS (existing `test_strict_grounding` must still pass).
- [ ] **Step 5:** Commit: `feat(rag): numbered source/page citations in context, prompt and UI`.

> [!NOTE]
> The current LoRA model was fine-tuned on the old context format. It will usually follow the new prompt, but citation compliance improves after re-running `pd-train-llm` (optional; not part of this plan).

---

### Task 8: Rebuild, verify, document

**Files:**
- Modify: `README.md` (RAG section: hybrid retrieval, reranker download, `pd-eval-rag`, golden format)

- [ ] **Step 1:** `ruff check .` and `pytest -q` → both clean.
- [ ] **Step 2:** If `data/knowledge_base/` contains user documents: back up `vector_store/faiss_index` to `vector_store/faiss_index.bak`, run `pd-build-index`, then check `index.ntotal > 0` and that every doc has `chunk_uid`. If the KB is empty, skip this step and keep the existing index (the legacy path is covered by Task 4).
- [ ] **Step 3:** If a golden set exists, run `pd-eval-rag` and paste the baseline, hybrid and rerank tables into the README.
- [ ] **Step 4:** Streamlit smoke test: upload a small `.md`, ask a question about it, and confirm the answer contains `[1]` and the expander marks that source as cited. Restart the app and confirm the doc is still retrievable. Delete it and confirm it is no longer retrieved.
- [ ] **Step 5:** Commit: `docs: hybrid RAG, reranker and evaluation`.
