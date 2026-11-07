"""Fine-tune LLM (mặc định Qwen2.5-1.5B-Instruct) bằng LoRA, gộp trọng số, đánh giá trên tập test.

    pd train llm                                  # cấu hình: configs/llm.yaml
    pd train llm train.epochs=1 eval.max_samples=20

Dữ liệu chia train/val/test theo bệnh (data/splits.py). Chỉ tính loss trên phần câu trả lời.
Sau khi train: lưu adapter LoRA -> gộp vào mô hình gốc (fp16) -> sinh câu trả lời cho tập test
và chấm ROUGE-L / token-F1 / chỉ số từ chối (evaluation/llm.py).
"""
from __future__ import annotations

import gc
import random
from pathlib import Path

from plant_doctor.data.llm_samples import build_samples
from plant_doctor.data.splits import group_split
from plant_doctor.utils.experiment import Run, metrics_table, set_seed


def make_tokenize_fn(tok, max_length: int):
    from plant_doctor.inference.prompts import build_messages

    def tokenize(sample):
        msgs = build_messages(sample["context"], sample["question"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": sample["answer"]}], tokenize=False)
        if not full.startswith(prompt):
            full = prompt + sample["answer"] + tok.eos_token
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = [-100] * n_prompt + ids[n_prompt:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels[: len(ids)]}
    return tokenize


def split_samples(cfg: dict) -> dict[str, list[dict]]:
    d = cfg["data"]
    samples = build_samples(rng=random.Random(cfg["seed"]), n_distractors=d.get("n_distractors", 1))
    return group_split(samples, group_fn=lambda s: s["group"], val=d["val_ratio"], test=d["test_ratio"],
                       seed=cfg["seed"])


def _tokenized(samples: list[dict], tok, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_list([{k: s[k] for k in ("context", "question", "answer")} for s in samples])
    ds = ds.map(make_tokenize_fn(tok, max_length), remove_columns=["context", "question", "answer"])
    return ds.filter(lambda x: any(lbl != -100 for lbl in x["labels"]))  # bỏ mẫu bị cắt mất câu trả lời


def _llm_report(metrics: dict, sizes: dict) -> str:
    return ("# Đánh giá LLM LoRA – tập test (bệnh chưa thấy khi train)\n\n"
            "Số mẫu: " + ", ".join(f"{k}={v}" for k, v in sizes.items()) + "\n\n"
            + metrics_table(metrics) + "\n\nChi tiết từng mẫu: `predictions.jsonl`.\n")


def _evaluate_and_log(run: Run, model, tok, test: list[dict], cfg: dict, sizes: dict) -> dict:
    from plant_doctor.evaluation.llm import evaluate_llm, make_generator

    if not test:
        print("⚠️ Tập test rỗng – bỏ qua đánh giá.")
        return {}
    metrics, rows = evaluate_llm(make_generator(model, tok, cfg["eval"]["max_new_tokens"]), test,
                                 cfg["eval"]["max_samples"])
    run.log_metrics(metrics, prefix="test")
    import json
    (run.dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    run.write_report(_llm_report(metrics, sizes))
    print(f"[test] ROUGE-L={metrics['rouge_l']:.4f} token-F1={metrics['token_f1']:.4f} "
          f"refusal-F1={metrics['refusal_f1']:.4f} false-refusal={metrics['false_refusal_rate']:.2%}")
    return metrics


def train_llm(cfg: dict) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    set_seed(cfg["seed"])
    m, t, lc = cfg["model"], cfg["train"], cfg["lora"]
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    split = split_samples(cfg)
    sizes = {k: len(v) for k, v in split.items()}
    print(f"Số mẫu: {sizes}")
    if not split["train"]:
        print("⚠️ Không có mẫu huấn luyện. Hãy đặt tài liệu vào data/knowledge_base/.")
        return {}

    run = Run.create("llm", cfg)
    tok = AutoTokenizer.from_pretrained(m["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = _tokenized(split["train"], tok, t["max_length"])
    val_ds = _tokenized(split["val"], tok, t["max_length"]) if split["val"] else None

    # Trọng số gốc fp32 + autocast fp16/bf16: tránh lỗi "unscale FP16 gradients" trên T4
    model = AutoModelForCausalLM.from_pretrained(m["base"], dtype=torch.float32,
                                                 device_map="auto" if cuda else None)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=lc["target_modules"]))
    model.print_trainable_parameters()

    Path(m["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(m["checkpoint_dir"]), num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch"], per_device_eval_batch_size=t["batch"],
        gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"], weight_decay=t["weight_decay"],
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=5,
        eval_strategy="epoch" if val_ds is not None else "no", save_strategy="no",
        fp16=cuda and not bf16, bf16=bf16, report_to="none", seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                                           pad_to_multiple_of=8))
    train_out = trainer.train()
    run.log_metrics({"train_loss": train_out.training_loss})
    run.save_json("train_log.json", trainer.state.log_history)

    adapter = Path(m["adapter_dir"])
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    print(f"Đã lưu adapter LoRA: {adapter}")

    del trainer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(m["base"], dtype=torch.float16 if cuda else torch.float32,
                                                device_map="auto" if cuda else None)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()
    out = Path(m["merged_dir"])
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"Đã lưu mô hình LLM (đã gộp LoRA) tại: {out}")

    metrics = _evaluate_and_log(run, merged, tok, split["test"], cfg, sizes)
    run.finish()
    return metrics


def evaluate_llm_run(cfg: dict, model_dir: str | None = None) -> dict:
    """Chỉ đánh giá mô hình đã gộp (model.merged_dir hoặc model_dir) trên tập test."""
    from plant_doctor.evaluation.llm import load_model

    set_seed(cfg["seed"])
    split = split_samples(cfg)
    model_dir = model_dir or str(cfg["model"]["merged_dir"])
    run = Run.create("llm_eval", {**cfg, "evaluated_model": model_dir})
    model, tok = load_model(model_dir)
    metrics = _evaluate_and_log(run, model, tok, split["test"], cfg, {k: len(v) for k, v in split.items()})
    run.finish()
    return metrics
