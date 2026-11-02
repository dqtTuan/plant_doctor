# Dependency Upgrade & Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Plant Doctor onto current, bounded dependency versions (LangChain 1.x, transformers 5.22, sentence-transformers 6 trainer API, Streamlit 1.66) with lint + tests green locally and in CI.

**Architecture:** `pyproject.toml` becomes the single source of truth for dependencies (with `<next-major` bounds); `requirements.txt` only installs the package. Deprecated call sites are migrated one library at a time, each pinned by a regression test. CI installs CPU torch + the package so the real test suite runs.

**Tech Stack:** Python 3.12, setuptools, pytest 9, ruff, LangChain 1.x, transformers 5, sentence-transformers 6, Streamlit 1.66, GitHub Actions, Docker.

**Spec:** `C:\Users\tuanas\.gemini\antigravity\brain\78420005-e058-4c75-8470-1ef9c27c76c9\upgrade_audit.md` (findings D1–D7, phases P0 + P1).

## Global Constraints

- Python floor stays `>=3.10`; ruff `target-version = "py310"`.
- App stays 100% offline at runtime (`HF_HUB_OFFLINE=1` is preserved).
- Every runtime dependency in `pyproject.toml` has both a floor and an upper bound (`<next major`, or `<0.(minor+1)` for 0.x packages).
- No changes to model weights, `vector_store/`, or `data/knowledge_base/`. Do **not** run `pd-build-index` (the KB folder is empty; it would overwrite the existing index).
- Full `pytest -q` must pass at the end of every task.

## Review Focus

1. Installing into an env that already has `langchain==0.3.x` (meta package) next to `langchain-core 1.x`: pip must not leave a conflicting install → Task 2 uninstalls/excludes it and `pip check` must be clean.
2. Existing FAISS index (`index.pkl`, pickled with LangChain 0.3 `Document`) must still load under LangChain 1.x → Task 3 round-trip test + manual load of the real index.
3. Docker build after `requirements.txt` stops listing packages → Task 2 changes the Dockerfile layer order; `docker build` must succeed.
4. CI runner without GPU → tests must pass with CPU torch only (Task 1).
5. Streamlit app boots with the real models present → Task 5 smoke run.

---

### Task 1: Dev tooling, lint, and CI that actually runs the tests

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Modify: `.github/workflows/ci.yml` (job `lint-test`)
- Modify: any file flagged by `ruff check .`

**Interfaces:**
- Produces: `pip install -e .[dev]` installs `pytest>=8,<10`, `pytest-cov>=5`, `ruff>=0.6,<1`.

- [ ] **Step 1:** Set `dev = ["pytest>=8,<10", "pytest-cov>=5", "ruff>=0.6,<1"]` and run `pip install -e .[dev]`.
- [ ] **Step 2:** Run `ruff check .` — record the violation count.
- [ ] **Step 3:** Fix violations (`ruff check --fix .`, then manual fixes; e.g. unused `e` in `except Exception as e: pass` in `pipeline.py`, unused imports in tests). Do not change behavior.
- [ ] **Step 4:** Run `ruff check .` → Expected: `All checks passed!`; `pytest -q` → all pass.
- [ ] **Step 5:** In `ci.yml` replace `pip install ruff pytest` with:
  ```yaml
  - name: Install
    run: |
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      pip install -e .[dev]
  ```
  Remove the three blank lines after the Ruff step.
- [ ] **Step 6:** Commit: `ci: install package + CPU torch so tests run; add ruff to dev extra; fix lint`.

---

### Task 2: Single source of truth + bounded versions

**Files:**
- Modify: `pyproject.toml` (`dependencies`, new `pg` extra)
- Modify: `requirements.txt`
- Modify: `Dockerfile` (dependency install layer)
- Test: `tests/test_packaging.py` (create)

**Interfaces:**
- Produces: extras `dev`, `pg`. Later plans add deps only to `pyproject.toml`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_packaging.py
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _deps():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]

def test_every_runtime_dependency_has_upper_bound():
    unbounded = [d for d in _deps() if "<" not in d]
    assert unbounded == []

def test_langchain_is_1x():
    deps = " ".join(_deps())
    assert "langchain-core>=1.7,<2" in deps
    assert "langchain-community>=0.4,<0.5" in deps

def test_requirements_txt_only_installs_package():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pkgs = [l for l in lines if l and not l.startswith("#")]
    assert pkgs == ["-e ."]
```

- [ ] **Step 2:** `pytest tests/test_packaging.py -v` → Expected: 3 FAIL.
- [ ] **Step 3:** Set `dependencies` to exactly:
  `ultralytics>=8.4.198,<9`, `transformers>=5.22,<6`, `peft>=0.21,<1`, `accelerate>=1.15,<2`, `datasets>=5.1,<6`, `sentence-transformers>=6.2,<7`, `langchain-core>=1.7,<2`, `langchain-community>=0.4,<0.5`, `langchain-huggingface>=1.2,<2`, `langchain-text-splitters>=1.1,<2`, `faiss-cpu>=1.15,<2`, `pypdf>=6.21,<7`, `opencv-python-headless>=5.0,<6`, `pyyaml>=6,<7`, `pillow>=11,<13`, `tqdm>=4.66,<5`, `streamlit>=1.66,<2`, `requests>=2.32.4,<3`.
  Add `pg = ["psycopg2-binary>=2.9,<3", "pgvector>=0.3,<1"]`. Keep the torch comment.
  `requirements.txt` → the torch comment + `-e .` only.
- [ ] **Step 4:** Dockerfile: replace the `COPY requirements.txt` / `grep -v` layer with `COPY pyproject.toml README.md ./` + `COPY src ./src` + `pip install .` (cache mount kept), and delete the later duplicate `pip install --no-deps` step.
- [ ] **Step 5:** `pip uninstall -y langchain` (unused 0.3 meta package), then `pip install -e .[dev] -U`; `pip check` → Expected: `No broken requirements found.`
- [ ] **Step 6:** `pytest -q` → Expected: all pass (including the 3 new tests). If LangChain 1.x breaks imports, fix them in Task 3 before committing both together.
- [ ] **Step 7:** `docker build -t plant-doctor:dev .` → Expected: success. (Skip with a note if Docker isn't available locally; CI covers it.)
- [ ] **Step 8:** Commit: `build: bounded deps in pyproject as single source; LangChain 1.x`.

---

### Task 3: LangChain 1.x compatibility

**Files:**
- Modify (only if imports break): `src/plant_doctor/pipeline.py`, `src/plant_doctor/rag/build_index.py`, `src/plant_doctor/rag/semantic_chunking.py`, `src/plant_doctor/rag/document_manager.py`
- Test: `tests/test_langchain_compat.py` (create)

**Interfaces:**
- Consumes: current imports `langchain_core.documents.Document`, `langchain_community.vectorstores.FAISS`, `langchain_community.vectorstores.utils.DistanceStrategy`, `langchain_huggingface.HuggingFaceEmbeddings`, `langchain_text_splitters.RecursiveCharacterTextSplitter`.

- [ ] **Step 1: Write the test**

```python
# tests/test_langchain_compat.py
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
import numpy as np
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
    store = FAISS.load_local(str(tmp_path), emb, allow_dangerous_deserialization=True,
                             distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    hit = store.similarity_search("đạo ôn lúa", k=1)[0]
    assert hit.page_content == "đạo ôn lúa"
    assert hit.metadata["source"] == "a.md"
```

- [ ] **Step 2:** `pytest tests/test_langchain_compat.py -v` → Expected: PASS on 1.x (if FAIL, fix the import path it names).
- [ ] **Step 3:** Verify the real index loads:
  `python -c "from plant_doctor.pipeline import PlantDoctorPipeline as P; p=P(load_models=False); print(type(p.load_vector_store()).__name__, p.vector_store.index.ntotal)"` → Expected: `FAISS <n>` with n > 0.
- [ ] **Step 4:** `pytest -q` → all pass. Commit: `test: pin FAISS round-trip under LangChain 1.x`.

---

### Task 4: transformers `dtype`, Streamlit `width`, sentence-transformers trainer

**Files:**
- Modify: `src/plant_doctor/pipeline.py` (`load_llm`), `src/plant_doctor/training/train_llm.py` (2 `from_pretrained` calls)
- Modify: `app/streamlit_app.py` (5 × `use_container_width=True` → `width="stretch"`)
- Modify: `src/plant_doctor/training/train_embedding.py` (`train_custom_embedding`)
- Test: `tests/test_deprecations.py` (create)

**Interfaces:**
- Produces: `build_train_dataset(pairs: list[tuple[str, str]]) -> datasets.Dataset` in `train_embedding.py`, columns `anchor`, `positive`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deprecations.py
import importlib
import warnings
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]

def test_load_llm_uses_dtype_kwarg(tmp_path, monkeypatch):
    import plant_doctor.pipeline as pl
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(pl, "LLM_MERGED_DIR", tmp_path)
    fake_model = MagicMock()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: MagicMock())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_model)
    pl.PlantDoctorPipeline(load_models=False).load_llm()
    kwargs = fake_model.call_args.kwargs
    assert "dtype" in kwargs and "torch_dtype" not in kwargs

def test_no_torch_dtype_or_use_container_width_in_sources():
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "app" / "streamlit_app.py"]
    offenders = [f.name for f in files
                 if "torch_dtype=" in f.read_text(encoding="utf-8")
                 or "use_container_width" in f.read_text(encoding="utf-8")]
    assert offenders == []

def test_train_embedding_imports_without_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(importlib.import_module("plant_doctor.training.train_embedding"))

def test_build_train_dataset_columns():
    from plant_doctor.training.train_embedding import build_train_dataset
    ds = build_train_dataset([("q1", "p1"), ("q2", "p2")])
    assert ds.column_names == ["anchor", "positive"]
    assert len(ds) == 2
```

- [ ] **Step 2:** `pytest tests/test_deprecations.py -v` → Expected: 4 FAIL.
- [ ] **Step 3:** Replace `torch_dtype=` with `dtype=` (3 sites) and `use_container_width=True` with `width="stretch"` (5 sites).
- [ ] **Step 4:** In `train_embedding.py`: drop `InputExample`, `NoDuplicatesDataLoader`, `model.fit`. Import from `sentence_transformers`: `SentenceTransformer`, `SentenceTransformerTrainer`, `SentenceTransformerTrainingArguments`; from `sentence_transformers.sentence_transformer.training_args`: `BatchSamplers` (the top-level `training_args` path is the deprecated alias); loss from `sentence_transformers.sentence_transformer.losses`. Train with `SentenceTransformerTrainingArguments(output_dir=str(MODELS_DIR / "embedding_checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch, batch_sampler=BatchSamplers.NO_DUPLICATES, seed=SEED, save_strategy="no", report_to="none")` and `MultipleNegativesRankingLoss`; keep `top1_accuracy` before/after and `model.save(str(output_dir))`.
- [ ] **Step 5:** `pytest -q` → all pass, **0 warnings from plant_doctor modules**.
- [ ] **Step 6:** Commit: `refactor: migrate to transformers dtype, streamlit width, ST trainer API`.

---

### Task 5: End-to-end verification

**Files:** none (verification only; fix regressions in the owning file if found)

- [ ] **Step 1:** `ruff check .` → `All checks passed!`; `pytest -q` → all pass; `pip check` → clean.
- [ ] **Step 2:** Load real models: `python -c "from plant_doctor.pipeline import PlantDoctorPipeline as P; p=P(); print(bool(p.yolo), bool(p.vector_store), bool(p.llm))"` → Expected: `True True True`.
- [ ] **Step 3:** Generate once: build messages with `build_messages("(Không tìm thấy tài liệu phù hợp)", "Xin chào")` and print `"".join(p.stream_answer(msgs, max_new_tokens=32))` → Expected: non-empty Vietnamese text, no exceptions.
- [ ] **Step 4:** `streamlit run app/streamlit_app.py --server.headless=true` (background); `curl http://localhost:8501/_stcore/health` → `ok`; stop the server.
- [ ] **Step 5:** Update README install section (`pip install -e .[dev]`, `.[pg]` for PostgreSQL). Commit: `docs: install instructions for extras`.
