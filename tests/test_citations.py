from langchain_core.documents import Document

from plant_doctor.prompts import SYSTEM_PROMPT, extract_citations, format_context


def test_format_context_includes_source_and_page():
    ctx = format_context([
        Document(page_content="Phun Trichoderma.", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="Tỉa cành.", metadata={"source": "b.md"}),
    ])
    assert ctx.startswith("[1] (a.pdf, trang 3)\nPhun Trichoderma.")
    assert "[2] (b.md)\nTỉa cành." in ctx


def test_format_context_empty_unchanged():
    assert format_context([]) == "(Không tìm thấy tài liệu phù hợp)"


def test_extract_citations_filters_and_dedupes():
    assert extract_citations("Phun sớm [2][1]. Lặp [2]. Sai [7].", n_docs=3) == [2, 1]


def test_prompt_requires_citations():
    assert "[1]" in SYSTEM_PROMPT and "Trích dẫn" in SYSTEM_PROMPT
