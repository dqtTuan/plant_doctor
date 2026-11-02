from pathlib import Path
from plant_doctor.data.scrape_web import (
    extract_html_to_markdown,
    save_markdown_file,
    slugify_vietnamese,
)


def test_slugify_vietnamese():
    assert slugify_vietnamese("Bệnh Đạo Ôn Trên Cây Lúa") == "benh_dao_on_tren_cay_lua"
    assert slugify_vietnamese("Sâu Vẽ Bùa & Bọ Trĩ (2026)") == "sau_ve_bua_bo_tri_2026"
    assert slugify_vietnamese("Đặc điểm & Phòng trừ") == "dac_diem_phong_tru"
    assert slugify_vietnamese("") == "tai_lieu_web"


def test_extract_html_to_markdown():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cách trị bệnh phấn trắng trên dưa leo</title>
        <meta property="og:title" content="Cách trị bệnh phấn trắng trên dưa leo hiệu quả" />
    </head>
    <body>
        <header>
            <nav class="navbar"><a href="/">Trang chủ</a></nav>
        </header>
        <div class="advertisement">Mua phân bón giảm giá 50%</div>
        <article class="article-content">
            <h2>1. Triệu chứng nhận biết</h2>
            <p>Bệnh phấn trắng xuất hiện vết đốm trắng mốc như bột phấn trên bề mặt lá.</p>
            <table>
                <tr><th>Thuốc đặc trị</th><th>Liều lượng</th></tr>
                <tr><td>Anvil 5SC</td><td>1ml / 1 lít nước</td></tr>
            </table>
        </article>
        <aside class="sidebar">Tin mới nhất...</aside>
        <footer>Bản quyền 2026</footer>
    </body>
    </html>
    """

    title, md = extract_html_to_markdown(sample_html, url="https://example.com/phan-trang")

    assert title == "Cách trị bệnh phấn trắng trên dưa leo hiệu quả"
    assert "https://example.com/phan-trang" in md
    assert "Triệu chứng nhận biết" in md
    assert "Anvil 5SC" in md
    # Make sure noise elements were stripped
    assert "Trang chủ" not in md
    assert "Mua phân bón giảm giá 50%" not in md
    assert "Tin mới nhất" not in md
    assert "Bản quyền" not in md


def test_save_markdown_file(tmp_path: Path):
    sample_content = "# Bệnh Đạo Ôn\nNội dung hướng dẫn phòng trừ..."
    title = "Bệnh Đạo Ôn Trên Lúa"

    saved_file = save_markdown_file(sample_content, title, out_dir=tmp_path)
    assert saved_file.exists()
    assert saved_file.name == "benh_dao_on_tren_lua.md"
    assert saved_file.read_text(encoding="utf-8") == sample_content

    # Custom filename
    custom_saved = save_markdown_file(
        sample_content, title, out_dir=tmp_path, out_file="custom_doc.md"
    )
    assert custom_saved.exists()
    assert custom_saved.name == "custom_doc.md"
