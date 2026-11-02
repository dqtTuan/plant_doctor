#!/usr/bin/env python3
"""Runner script: Cào nội dung bài viết từ website nông nghiệp sang định dạng Markdown.

Cách dùng:
  python scripts/scrape_web.py https://nongnghiep.vn/...
  python scripts/scrape_web.py --url https://... --out-file benh_lua.md
  python scripts/scrape_web.py --file urls.txt
"""
import sys
from pathlib import Path

# Đảm bảo import được module plant_doctor từ thư mục src
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plant_doctor.data.scrape_web import main  # noqa: E402

if __name__ == "__main__":
    main()
