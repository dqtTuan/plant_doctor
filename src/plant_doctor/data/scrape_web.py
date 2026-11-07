"""Công cụ tìm kiếm và cào dữ liệu từ trang web sang định dạng Markdown chuẩn cho RAG.

Sử dụng:
  pd data scrape --query "bệnh sương mai cà chua cách trị" --top 10
  pd data scrape -q "bệnh đạo ôn lúa" -q "bệnh thán thư ớt" --top 10
  pd data scrape --url "https://nongnghiep.vn/benh-dao-on-lua..."
  pd data scrape "https://example.com/bai-1" "https://example.com/bai-2"
  pd data scrape --file urls.txt --out-dir data/knowledge_base/
"""
import argparse
from datetime import datetime
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote

from bs4 import BeautifulSoup
import html2text
import requests

from plant_doctor.config import KB_DIR

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

BLOCKED_DOMAINS = (
    "yahoo.",
    "bing.",
    "google.",
    "microsoft.",
    "uservoice.",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "fb.watch",
    "tiktok.com",
    "instagram.com",
    "shopee.vn",
    "lazada.vn",
    "tiki.vn",
    "sendo.vn",
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


FOOTER_CUTOFF_PATTERNS = [
    r"(?im)^#{0,4}\s*(?:bình luận|hãy đăng nhập để bình luận|để lại bình luận|ý kiến bạn đọc)\b",
    r"(?im)^#{0,4}\s*(?:bài viết liên quan|tin liên quan|bài viết cùng chuyên mục|có thể bạn quan tâm|bài viết mới nhất)\b",
    r"(?im)^#{0,4}\s*(?:sản phẩm đặc trị|sản phẩm liên quan|hot products|recommended news|hot news)\b",
    r"(?im)^#{0,4}\s*tell us about your agrochemical project\b",
    r"(?i)\b(?:share to|chia sẻ bài viết|chia sẻ):\s*(?:facebook|\[)",
    r"(?i)\bpost\s+post\s+hot\s+products\b",
    r"(?im)^copyright\s*©",
]

NOISE_LINE_PATTERNS = [
    r"(?i)^kích thước chữ$",
    r"(?i)^skip to content$",
    r"(?i)^table of contents$",
    r"(?i)^go to top$",
    r"(?i)^(?:xem thêm|đọc thêm|tham khảo thêm)\s*:.*$",
    r"(?i).*freeship.*\d+[\.,]\d+\s*₫.*",
    r"(?i)^link to #.*$",
    r"(?i)^link to mailto:.*$",
]


def clean_markdown_content(md_text: str) -> str:
    """Làm sạch sâu văn bản Markdown cào từ web để tối ưu cho Semantic Chunking & RAG."""
    if not md_text or not md_text.strip():
        return ""

    # 1. Tách riêng phần Header chuẩn (Tiêu đề + Nguồn tham khảo + ---) nếu có
    header_block = ""
    body = md_text
    sep_match = re.search(r"^---\s*$", md_text, flags=re.MULTILINE)
    if sep_match and sep_match.start() < 600:
        header_block = md_text[: sep_match.end()].strip() + "\n\n"
        body = md_text[sep_match.end() :]

    # 2. Xóa các khối JSON-LD / Schema.org và YAML frontmatter bị lọt vào nội dung
    body = re.sub(r"```json\s*\{\s*\"@context\".*?```", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"\{\s*\"@context\"\s*:\s*\"https?:\\?/\\?/schema\.org\".*", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"\\?---\s*description:\s*\".*?\\?---", "", body, flags=re.DOTALL | re.IGNORECASE)

    # 3. Xóa WordPress / PageBuilder shortcodes ([fusion_...], [vc_...], [/fusion_...])
    body = re.sub(r"\[/?(?:fusion_|vc_|elementor|et_pb_|rev_slider)[^\]]*\]", "", body, flags=re.IGNORECASE)

    # 4. Xóa các thẻ HTML đóng/mở bị escape hoặc còn sót (<\/h2>, <\/strong>, <\/li>, <div>...)
    body = re.sub(r"<\\?/[a-zA-Z0-9]+\s*>", "", body)
    body = re.sub(r"<[a-zA-Z0-9]+(?:\s+[^>]*)?>", "", body)

    # 5. Xóa hình ảnh Markdown và hình ảnh gắn link: [![alt](img)](link) và ![alt](img)
    body = re.sub(r"\[\s*!\[[^\]]*\]\([^)]*\)\s*\]\([^)]*\)", "", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)

    # 6. Chuyển liên kết [text](url "title") thành text thuần (tránh làm nhiễu vector embedding)
    def _unwrap_link(match: re.Match) -> str:
        label = match.group(1).strip()
        if not label or label.lower().startswith(("http://", "https://", "link to ", "skip to")):
            return ""
        return label

    body = re.sub(r"\[([^\]]*)\]\((?:https?:|mailto:|/|#)[^)]*\)", _unwrap_link, body)

    # 7. Đưa các tiêu đề Markdown (##, ###) và gạch đầu dòng (\-) bị dính liền trên cùng 1 dòng xuống dòng mới
    body = re.sub(r"(?<!\n)(?<!#)\s+(#{1,4}\s+[A-ZÀ-Ỹ0-9])", r"\n\n\1", body)
    body = re.sub(r"\s+\\-\s*", "\n- ", body)

    # 8. Cắt bỏ phần đuôi rác (Bình luận, Bài viết liên quan, Quảng cáo cuối bài, Footer)
    earliest_cut = len(body)
    for pat in FOOTER_CUTOFF_PATTERNS:
        m = re.search(pat, body)
        if m and m.start() > 150 and m.start() < earliest_cut:
            earliest_cut = m.start()
    body = body[:earliest_cut]

    # 9. Làm sạch từng dòng
    cleaned_lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.replace("\r", "").strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Chuẩn hóa ký tự gạch đầu dòng bị escape (\- -> -, 1\. -> 1.)
        line = re.sub(r"^\\-\s*", "- ", line)
        line = re.sub(r"^(\d+)\\\.\s*", r"\1. ", line)
        line = re.sub(r"\*\*([^\*\n]+?)\s+\*\*", r"**\1**", line)
        line = re.sub(r"([a-zA-ZÀ-ỹ0-9,])(\*\*[A-ZÀ-Ỹa-z0-9])", r"\1 \2", line)

        # Bỏ các tiêu đề rỗng (chỉ có #, ##, ###)
        if re.match(r"^#{1,6}\s*$", line):
            continue

        # Bỏ các dòng chỉ toàn ký tự đặc biệt/dấu sao/dấu gạch (****, **__**, ---|---|---])
        if re.match(r"^[\s\*_\\/\-\|<>\"'\[\]\(\)\.]+$", line) and line != "---":
            continue

        # Bỏ các dòng điều hướng / quảng cáo rác
        if any(re.match(np, line) for np in NOISE_LINE_PATTERNS):
            continue

        cleaned_lines.append(line)

    cleaned_body = "\n".join(cleaned_lines)
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body).strip()
    return (header_block + cleaned_body).strip() + "\n"


def markdown_body_length(md_text: str) -> int:
    """Đếm số ký tự thực tế của phần nội dung bài viết (không tính phần header nguồn)."""
    sep_match = re.search(r"^---\s*$", md_text, flags=re.MULTILINE)
    body = md_text[sep_match.end() :] if (sep_match and sep_match.start() < 600) else md_text
    return len(body.strip())


def clean_knowledge_base(
    kb_dir: Path | str = KB_DIR,
    min_body_chars: int = 300,
    remove_short: bool = True,
) -> dict[str, int]:
    """Làm sạch toàn bộ các file .md trong thư mục tri thức và loại bỏ file rỗng/rác."""
    kb_path = Path(kb_dir)
    stats = {"total": 0, "cleaned": 0, "removed_short": 0, "removed_duplicate": 0, "bytes_saved": 0}
    if not kb_path.exists():
        return stats

    seen_bodies: set[str] = set()
    for md_file in sorted(kb_path.glob("*.md")):
        if md_file.name.lower().startswith("readme"):
            continue
        stats["total"] += 1
        try:
            raw_text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        orig_bytes = len(raw_text.encode("utf-8"))
        cleaned_text = clean_markdown_content(raw_text)
        body_len = markdown_body_length(cleaned_text)

        # Xóa các file không có nội dung hoặc quá ngắn (trang danh mục, lỗi tải)
        if body_len < min_body_chars:
            if remove_short:
                md_file.unlink(missing_ok=True)
                stats["removed_short"] += 1
                stats["bytes_saved"] += orig_bytes
            continue

        # Kiểm tra trùng lặp nội dung chính
        sep_match = re.search(r"^---\s*$", cleaned_text, flags=re.MULTILINE)
        body_only = cleaned_text[sep_match.end() :].strip() if sep_match else cleaned_text.strip()
        body_fingerprint = re.sub(r"\s+", " ", body_only[:1000]).lower()
        if body_fingerprint in seen_bodies:
            if remove_short:
                md_file.unlink(missing_ok=True)
                stats["removed_duplicate"] += 1
                stats["bytes_saved"] += orig_bytes
            continue
        seen_bodies.add(body_fingerprint)

        new_bytes = len(cleaned_text.encode("utf-8"))
        if cleaned_text != raw_text:
            md_file.write_text(cleaned_text, encoding="utf-8")
            stats["cleaned"] += 1
            stats["bytes_saved"] += max(0, orig_bytes - new_bytes)

    return stats


def extract_html_to_markdown(html_content: str, url: str = "") -> tuple[str, str]:
    """Phân tích HTML, loại bỏ phần tử thừa và chuyển thành Markdown chuẩn."""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Trích xuất tiêu đề
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and hasattr(og_title, "get"):
        content_val = og_title.get("content")
        if isinstance(content_val, list):
            title = " ".join(str(item) for item in content_val).strip()
        elif isinstance(content_val, str):
            title = content_val.strip()

    if not title:
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text().strip()
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
    return title, clean_markdown_content(full_markdown)


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


_SEARCH_SESSION: requests.Session | None = None


def _get_search_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    global _SEARCH_SESSION
    if _SEARCH_SESSION is None:
        _SEARCH_SESSION = requests.Session()
        _SEARCH_SESSION.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
    return _SEARCH_SESSION


def search_web_urls(
    query: str,
    max_results: int = 10,
    timeout: int = 15,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[str]:
    """Tìm kiếm từ khóa trên web và trích xuất danh sách N URL kết quả đầu tiên."""
    session = _get_search_session(user_agent)
    urls: list[str] = []
    # Lấy dư ứng viên (tối đa 4 trang tìm kiếm) phòng trường hợp một số trang chặn cào
    target_candidates = max(max_results * 2, 15)

    for offset in (1, 8, 15, 22):
        params = {"p": query} if offset == 1 else {"p": query, "b": offset}
        html_text = ""
        # Thử tối đa 3 lần (lần 1 khởi tạo cookie session nếu chưa có)
        for _attempt in range(3):
            try:
                resp = session.get(
                    "https://search.yahoo.com/search",
                    params=params,
                    timeout=timeout,
                )
                if resp.status_code == 200 and len(resp.text) > 500:
                    html_text = resp.text
                    break
            except Exception:
                continue

        if not html_text:
            continue

        soup = BeautifulSoup(html_text, "html.parser")
        anchors = soup.select("div#web a[href]") or soup.find_all("a", href=True)
        for a in anchors:
            href = str(a.get("href", ""))
            m = re.search(r"/RU=([^/]+)/", href)
            candidate = unquote(m.group(1)) if m else href
            if not candidate.startswith(("http://", "https://")):
                continue
            cand_lower = candidate.lower()
            if any(domain in cand_lower for domain in BLOCKED_DOMAINS):
                continue
            if candidate not in urls:
                urls.append(candidate)
            if len(urls) >= target_candidates:
                return urls

    return urls


def search_and_scrape(
    query: str,
    top_k: int = 10,
    out_dir: Path | str = KB_DIR,
    timeout: int = 15,
    min_chars: int = 300,
) -> list[Path]:
    """Tìm kiếm từ khóa và cào đủ top_k trang web đầu tiên lưu thành Markdown."""
    print(f"\n🔍 Đang tìm kiếm: '{query}' (mục tiêu: {top_k} trang đầu tiên)...")
    candidate_urls = search_web_urls(query, max_results=top_k, timeout=timeout)
    if not candidate_urls:
        print(f"⚠️ Không tìm thấy kết quả nào cho từ khóa: '{query}'")
        return []

    saved_files: list[Path] = []
    for url in candidate_urls:
        if len(saved_files) >= top_k:
            break
        idx = len(saved_files) + 1
        print(f"  [{idx}/{top_k}] Đang tải: {url} ...", end=" ", flush=True)
        try:
            title, md_content = scrape_url(url, timeout=timeout)
            if markdown_body_length(md_content) < min_chars:
                print("⚠️ Bỏ qua (nội dung sau làm sạch quá ngắn)")
                continue
            saved_path = save_markdown_file(md_content, title, out_dir=out_dir)
            print(f"✅ Đã lưu -> {saved_path.name}")
            saved_files.append(saved_path)
        except Exception as err:
            print(f"❌ Lỗi ({type(err).__name__}) - chuyển sang trang tiếp theo")

    return saved_files


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="Tìm kiếm, cào bài viết và làm sạch dữ liệu Markdown nông nghiệp cho RAG."
    )
    ap.add_argument("urls", nargs="*", help="Danh sách URL cần cào (ngăn cách bởi dấu cách)")
    ap.add_argument("--url", "-u", action="append", help="URL cần cào (có thể truyền nhiều lần)")
    ap.add_argument("--file", "-f", help="File chứa danh sách URL (mỗi URL trên 1 dòng)")
    ap.add_argument(
        "--query",
        "-q",
        action="append",
        help="Từ khóa tìm kiếm bệnh cây trồng để tự động cào top trang đầu tiên (có thể truyền nhiều lần)",
    )
    ap.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        help="Số lượng trang web đầu tiên cần cào cho mỗi từ khóa tìm kiếm (mặc định: 10)",
    )
    ap.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="Làm sạch toàn bộ các file .md hiện có trong thư mục tri thức (xóa rác HTML, ảnh, quảng cáo, file rỗng)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Độ dài nội dung tối thiểu (ký tự) để giữ lại file sau khi làm sạch (mặc định: 300)",
    )
    ap.add_argument("--out-dir", "-o", default=str(KB_DIR), help="Thư mục lưu file Markdown (mặc định data/knowledge_base/)")
    ap.add_argument("--out-file", help="Tên file đầu ra (chỉ áp dụng khi cào 1 URL đơn lẻ)")
    ap.add_argument("--timeout", type=int, default=15, help="Thời gian chờ tối đa (giây)")
    args = ap.parse_args()

    # 1. Chế độ tìm kiếm từ khóa và tự động cào top N trang web
    queries = list(args.query or [])
    total_saved = 0
    if queries:
        for q in queries:
            saved = search_and_scrape(
                query=q,
                top_k=args.top,
                out_dir=args.out_dir,
                timeout=args.timeout,
                min_chars=args.min_chars,
            )
            total_saved += len(saved)

    # 2. Chế độ cào theo danh sách URL trực tiếp
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

    target_urls = list(dict.fromkeys(target_urls))

    if target_urls:
        print(f"\n🌐 Bắt đầu cào {len(target_urls)} trang web chỉ định vào thư mục: {args.out_dir}")
        for i, u in enumerate(target_urls, 1):
            print(f"[{i}/{len(target_urls)}] Đang tải: {u} ...", end=" ", flush=True)
            try:
                title, md_content = scrape_url(u, timeout=args.timeout)
                out_name = args.out_file if (len(target_urls) == 1 and args.out_file) else None
                saved_path = save_markdown_file(md_content, title, out_dir=args.out_dir, out_file=out_name)
                print(f"✅ Đã lưu -> {saved_path.name}")
                total_saved += 1
            except Exception as err:
                print(f"❌ Lỗi: {err}")

    # 3. Chế độ làm sạch dữ liệu trong thư mục knowledge_base
    if args.clean or queries or target_urls:
        print(f"\n🧹 Đang làm sạch và chuẩn hóa dữ liệu trong '{args.out_dir}'...")
        stats = clean_knowledge_base(kb_dir=args.out_dir, min_body_chars=args.min_chars)
        kb_saved = stats["bytes_saved"] / 1024.0
        print(
            f"   - Tổng số file kiểm tra : {stats['total']}\n"
            f"   - Đã lọc nhiễu & chuẩn hóa: {stats['cleaned']} file\n"
            f"   - Đã xóa file rỗng/ngắn  : {stats['removed_short']} file\n"
            f"   - Đã xóa file trùng lặp  : {stats['removed_duplicate']} file\n"
            f"   - Dung lượng rác đã dọn  : {kb_saved:.1f} KB"
        )
        if not queries and not target_urls:
            print("👉 Hãy chạy: `python -m plant_doctor rag build --chunker-type semantic` để cập nhật Vector DB!")
            return

    if not queries and not target_urls and not args.clean:
        print("❌ Vui lòng cung cấp từ khóa (--query), URL (--url) hoặc cờ làm sạch (--clean)!")
        print("Ví dụ:")
        print('  python scripts/scrape_web.py --clean')
        print('  python scripts/scrape_web.py --query "bệnh sương mai cà chua" --top 10')
        return

    print(f"\n🎉 Hoàn thành! Đã lưu thành công {total_saved} tài liệu vào '{args.out_dir}'.")
    print("👉 Hãy chạy: `python -m plant_doctor rag build --chunker-type semantic` để nạp ngay tài liệu vào Vector DB!")


if __name__ == "__main__":
    main()
