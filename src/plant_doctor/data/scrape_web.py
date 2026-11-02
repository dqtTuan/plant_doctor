"""Công cụ cào dữ liệu từ trang web và chuyển đổi sang định dạng Markdown chuẩn cho RAG.

Sử dụng:
  pd-scrape --url "https://nongnghiep.vn/benh-dao-on-lua..."
  pd-scrape "https://example.com/bai-1" "https://example.com/bai-2"
  pd-scrape --file urls.txt --out-dir data/knowledge_base/
"""
import argparse
from datetime import datetime
from pathlib import Path
import re
import unicodedata

from bs4 import BeautifulSoup
import html2text
import requests

from plant_doctor.config import KB_DIR

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

NOISE_TAGS = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
    "noscript",
]

NOISE_SELECTORS = [
    ".menu",
    ".nav",
    ".navbar",
    ".footer",
    ".header",
    ".sidebar",
    ".advertisement",
    ".ad",
    ".ads",
    ".social-share",
    ".comment",
    ".comments",
    ".relate-news",
    ".tin-lien-quan",
]

CONTENT_SELECTORS = [
    "article",
    "main",
    ".article-content",
    ".post-content",
    ".entry-content",
    ".detail-content",
    ".fck_detail",
    ".content-detail",
    ".story__content",
    "#article-body",
    "#content",
    ".content",
]


def slugify_vietnamese(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt thành tên file an toàn (slug)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("đ", "d").replace("Đ", "d")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", text)
    return slug[:80] if slug else "tai_lieu_web"


def extract_html_to_markdown(html_content: str, url: str = "") -> tuple[str, str]:
    """Phân tích HTML, loại bỏ phần tử thừa và chuyển thành Markdown chuẩn."""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Trích xuất tiêu đề
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text().strip()
    elif soup.title:
        title = soup.title.get_text().strip()
    title = title or "Tài liệu chuyên môn nông nghiệp"

    # 2. Xóa các thẻ rác, quảng cáo, menu
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    for selector in NOISE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    # 3. Tìm vùng nội dung chính
    main_el = None
    for selector in CONTENT_SELECTORS:
        found = soup.select_one(selector)
        if found and len(found.get_text().strip()) > 100:
            main_el = found
            break

    if main_el is None:
        main_el = soup.body or soup

    # 4. Cấu hình bộ chuyển đổi HTML -> Markdown
    converter = html2text.HTML2Text()
    converter.body_width = 0  # Không tự ngắt dòng cứng
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_emphasis = False
    converter.ignore_tables = False
    converter.single_line_break = False

    raw_md = converter.handle(str(main_el))

    # 5. Dọn dẹp khoảng trắng và định dạng Markdown
    cleaned_md = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

    # Đảm bảo tiêu đề chính ở đầu file
    today_str = datetime.now().strftime("%Y-%m-%d")
    header_parts = [f"# {title}\n"]
    if url:
        header_parts.append(f"> **Nguồn tham khảo**: [{url}]({url})  \n> **Ngày thu thập**: {today_str}\n\n---\n")

    full_markdown = "\n".join(header_parts) + "\n" + cleaned_md
    return title, full_markdown


def scrape_url(
    url: str,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[str, str]:
    """Tải và cào nội dung một trang web thành Markdown."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Tự động phát hiện encoding đúng của trang web tiếng Việt
    resp.encoding = resp.apparent_encoding or "utf-8"
    return extract_html_to_markdown(resp.text, url=url)


def save_markdown_file(
    content: str,
    title: str,
    out_dir: Path | str = KB_DIR,
    out_file: str | None = None,
) -> Path:
    """Lưu nội dung Markdown vào thư mục chỉ định (mặc định là data/knowledge_base/)."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if out_file:
        file_path = target_dir / (out_file if out_file.endswith(".md") else f"{out_file}.md")
    else:
        file_name = f"{slugify_vietnamese(title)}.md"
        file_path = target_dir / file_name

    file_path.write_text(content, encoding="utf-8")
    return file_path


def main():
    ap = argparse.ArgumentParser(description="Cào bài viết từ website nông nghiệp và lưu về định dạng Markdown cho RAG.")
    ap.add_argument("urls", nargs="*", help="Danh sách URL cần cào (ngăn cách bởi dấu cách)")
    ap.add_argument("--url", "-u", action="append", help="URL cần cào (có thể truyền nhiều lần)")
    ap.add_argument("--file", "-f", help="File chứa danh sách URL (mỗi URL trên 1 dòng)")
    ap.add_argument("--out-dir", "-o", default=str(KB_DIR), help="Thư mục lưu file Markdown (mặc định data/knowledge_base/)")
    ap.add_argument("--out-file", help="Tên file đầu ra (chỉ áp dụng khi cào 1 URL đơn lẻ)")
    ap.add_argument("--timeout", type=int, default=15, help="Thời gian chờ tối đa (giây)")
    args = ap.parse_args()

    target_urls = list(args.urls or [])
    if args.url:
        target_urls.extend(args.url)
    if args.file:
        p_file = Path(args.file)
        if p_file.exists():
            for line in p_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    target_urls.append(line)

    # Loại bỏ URL trùng lặp
    target_urls = list(dict.fromkeys(target_urls))

    if not target_urls:
        print("❌ Vui lòng cung cấp ít nhất một URL cần cào!")
        print("Ví dụ:")
        print("  pd-scrape --url \"https://nongnghiep.vn/...\"")
        print("  pd-scrape \"https://link1.com\" \"https://link2.com\"")
        return

    print(f"🌐 Bắt đầu cào {len(target_urls)} trang web vào thư mục: {args.out_dir}")
    success_count = 0
    for i, u in enumerate(target_urls, 1):
        print(f"[{i}/{len(target_urls)}] Đang tải: {u} ...", end=" ", flush=True)
        try:
            title, md_content = scrape_url(u, timeout=args.timeout)
            out_name = args.out_file if (len(target_urls) == 1 and args.out_file) else None
            saved_path = save_markdown_file(md_content, title, out_dir=args.out_dir, out_file=out_name)
            print(f"✅ Đã lưu -> {saved_path.name}")
            success_count += 1
        except Exception as err:
            print(f"❌ Lỗi: {err}")

    print(f"\n🎉 Hoàn thành! Đã cào thành công {success_count}/{len(target_urls)} tài liệu.")
    print("👉 Hãy chạy: `pd-build-index` để nạp ngay các tài liệu vừa cào vào Vector DB!")


if __name__ == "__main__":
    main()
