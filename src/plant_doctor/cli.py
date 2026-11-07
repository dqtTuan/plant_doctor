"""CLI hợp nhất cho Plant Doctor.

Cách dùng:
    pd train yolo [train.epochs=50 train.batch=32 ...]
    pd train embedding [train.epochs=10 ...]
    pd train llm [train.epochs=1 eval.max_samples=20 ...]

    pd eval yolo [eval.split=val ...]
    pd eval embedding
    pd eval llm [eval.max_samples=30 ...]
    pd eval rag [k=6 ...]
    pd eval all

    pd rag build [--chunker-type semantic ...]
    pd rag download-reranker

    pd data prepare-yolo [--src data/raw --method opencv ...]
    pd data scrape <url> [--out-dir data/knowledge_base ...]

    pd app [--port 8501]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from plant_doctor.config import ROOT, load_config


def _cmd_train(target: str, config_path: str | None, overrides: list[str]) -> None:
    cfg = load_config(config_path or target, overrides)
    if target == "yolo":
        from plant_doctor.training.yolo import train_yolo
        train_yolo(cfg)
    elif target == "embedding":
        from plant_doctor.training.embedding import train_embedding
        train_embedding(cfg)
    elif target == "llm":
        from plant_doctor.training.llm import train_llm
        train_llm(cfg)


def _cmd_eval(target: str, config_path: str | None, overrides: list[str]) -> None:
    if target == "all":
        for name in ("yolo", "embedding", "llm", "rag"):
            print(f"\n========== ĐÁNH GIÁ: {name.upper()} ==========")
            _cmd_eval(name, None, [])
        return

    cfg = load_config(config_path or target, overrides)
    if target == "yolo":
        from plant_doctor.training.yolo import evaluate_yolo_run
        evaluate_yolo_run(cfg)
    elif target == "embedding":
        from plant_doctor.training.embedding import evaluate_embedding_run
        evaluate_embedding_run(cfg)
    elif target == "llm":
        from plant_doctor.training.llm import evaluate_llm_run
        evaluate_llm_run(cfg)
    elif target == "rag":
        from plant_doctor.evaluation.rag import run_rag_eval
        run_rag_eval(cfg)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(
        prog="pd",
        description="Plant Doctor CLI – Huấn luyện, Đánh giá, RAG & Dữ liệu",
    )
    sub = ap.add_subparsers(dest="group", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Huấn luyện mô hình (yolo | embedding | llm)")
    p_train.add_argument("target", choices=["yolo", "embedding", "llm"])
    p_train.add_argument("-c", "--config", default=None, help="Đường dẫn file YAML cấu hình tùy chỉnh")
    p_train.add_argument("overrides", nargs="*", help="Ghi đè dạng key=value (VD: train.epochs=50)")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Đánh giá mô hình trên tập test (yolo | embedding | llm | rag | all)")
    p_eval.add_argument("target", choices=["yolo", "embedding", "llm", "rag", "all"])
    p_eval.add_argument("-c", "--config", default=None, help="Đường dẫn file YAML cấu hình tùy chỉnh")
    p_eval.add_argument("overrides", nargs="*", help="Ghi đè dạng key=value (VD: eval.split=val)")

    # --- rag ---
    p_rag = sub.add_parser("rag", help="Xây dựng chỉ mục FAISS/pgvector hoặc tải Reranker")
    p_rag.add_argument("action", choices=["build", "download-reranker"])
    p_rag.add_argument("extra", nargs=argparse.REMAINDER)

    # --- data ---
    p_data = sub.add_parser("data", help="Chuẩn bị dataset YOLO hoặc cào dữ liệu web")
    p_data.add_argument("action", choices=["prepare-yolo", "scrape"])
    p_data.add_argument("extra", nargs=argparse.REMAINDER)

    # --- app ---
    p_app = sub.add_parser("app", help="Khởi chạy giao diện Streamlit")
    p_app.add_argument("extra", nargs=argparse.REMAINDER)

    args = ap.parse_args(argv)

    if args.group == "train":
        _cmd_train(args.target, args.config, args.overrides)
    elif args.group == "eval":
        _cmd_eval(args.target, args.config, args.overrides)
    elif args.group == "rag":
        sys.argv = [f"pd rag {args.action}", *args.extra]
        if args.action == "build":
            from plant_doctor.rag.build_index import main as build_main
            build_main()
        else:
            from plant_doctor.rag.download_models import main as dl_main
            dl_main()
    elif args.group == "data":
        sys.argv = [f"pd data {args.action}", *args.extra]
        if args.action == "prepare-yolo":
            from plant_doctor.data.prepare_yolo import main as prep_main
            prep_main()
        else:
            from plant_doctor.data.scrape_web import main as scrape_main
            scrape_main()
    elif args.group == "app":
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), *args.extra]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

