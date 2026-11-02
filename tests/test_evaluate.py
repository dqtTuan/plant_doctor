from langchain_core.documents import Document
from plant_doctor.rag.evaluate import GoldenItem, evaluate, load_golden


def D(src, page=None):
    meta = {"source": src}
    if page is not None:
        meta["page"] = page
    return Document(page_content="x", metadata=meta)


def test_metrics_hit_mrr_and_no_answer():
    items = [
        GoldenItem("q1", [], [("a.pdf", 2)]),  # hit at rank 2
        GoldenItem("q2", [], [("b.md", None)]),  # miss
        GoldenItem("q3", [], []),  # unanswerable, retriever returns nothing
    ]
    table = {
        "q1": [D("z.md"), D("a.pdf", 2)],
        "q2": [D("c.md")],
        "q3": [],
    }
    m = evaluate(lambda q, labels: table[q], items, k=4)
    assert m["hit_at_k"] == 0.5
    assert m["mrr"] == 0.25
    assert m["no_answer_acc"] == 1.0
    assert (m["n_answerable"], m["n_unanswerable"]) == (2, 1)


def test_load_golden(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"question": "q", "labels": ["Tomato___Late_blight"], "expected": [{"source": "a.pdf", "page": 3}]}\n',
        encoding="utf-8",
    )
    assert load_golden(p) == [GoldenItem("q", ["Tomato___Late_blight"], [("a.pdf", 3)])]
