"""Bước 3: Fine-tune mô hình ngôn ngữ nhỏ (mặc định Qwen2.5-1.5B-Instruct) bằng LoRA.

Dữ liệu huấn luyện dạng RAG được sinh tự động từ data/knowledge_base/diseases.json:
  - Mẫu dương: ngữ cảnh chứa đúng bệnh (+ 1 tài liệu gây nhiễu) -> trả lời có cấu trúc
  - Mẫu âm  : ngữ cảnh KHÔNG chứa bệnh được hỏi -> từ chối (chống ảo giác)

python -m plant_doctor.training.train_llm --epochs 3
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

from plant_doctor.config import (LLM_ADAPTER_DIR, LLM_BASE, LLM_CHECKPOINT_DIR, LLM_MERGED_DIR,
                                 MODELS_DIR, SEED, TRAIN_DATA_DIR)
from plant_doctor.knowledge import entry_to_answer, entry_to_text, load_kb
from plant_doctor.prompts import build_messages, format_context

TEMPLATES = {
    "full": [
        "Mô hình YOLOv8 vừa phát hiện lá cây bị bệnh: {cls}. Hãy giải thích triệu chứng, "
        "nguyên nhân gây bệnh và cách điều trị sinh học/phòng ngừa.",
        "Cây {crop} nhà tôi bị {name}, tôi phải làm sao?",
        "Cho tôi phác đồ đầy đủ về {name} trên cây {crop}.",
    ],
    "symptoms": ["{name} trên {crop} có triệu chứng gì?", "Làm sao nhận biết {name} trên lá {crop}?"],
    "causes": ["Nguyên nhân nào gây ra {name} trên {crop}?", "Vì sao cây {crop} bị {name}?"],
    "treatment": [
        "Có cách nào chữa {name} trên {crop} mà hạn chế thuốc hóa học không?",
        "Phác đồ điều trị sinh học cho {name} trên {crop} là gì?",
    ],
    "prevention": ["Làm thế nào để phòng ngừa {name} cho cây {crop} ở vụ sau?"],
}

REFUSAL = (
    "Tài liệu tham khảo được cung cấp không chứa thông tin về {name} trên cây {crop}. "
    "Để tránh tư vấn sai, tôi không đưa ra phác đồ cụ thể. Bạn nên chụp thêm ảnh rõ nét hơn "
    "hoặc liên hệ cán bộ khuyến nông / trạm bảo vệ thực vật địa phương để được hỗ trợ."
)


def fill(tpl, e):
    cls = e["aliases"][0] if e.get("aliases") else f"{e['crop_en']} {e['name_en']}"
    return tpl.format(cls=cls, crop=e["crop_vi"].lower(), name=e["name_vi"].lower())


def extract_entries_from_markdown() -> list:
    """Trích xuất danh sách bệnh từ tài liệu Markdown trong kho tri thức."""
    from plant_doctor.config import KB_DIR
    import re
    entries = []
    for md_file in Path(KB_DIR).glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        blocks = re.split(r"\n##\s+", content)
        for block in blocks[1:]:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            title_line = lines[0]
            name_part = title_line.split("–")[0].strip() if "–" in title_line else title_line
            crop_part = title_line.split("–")[1].strip() if "–" in title_line else "Cây trồng"
            key = re.sub(r"[^0-9a-z]+", "_", name_part.lower()).strip("_")
            name_vi = name_part.split("(")[0].strip()
            name_en = name_part.split("(")[1].rstrip(")") if "(" in name_part else name_part
            crop_vi = crop_part.split(":")[1].split("(")[0].strip() if ":" in crop_part else crop_part
            crop_en = crop_part.split("(")[1].rstrip(")") if "(" in crop_part else crop_part
            entries.append({
                "key": key,
                "name_vi": name_vi,
                "name_en": name_en,
                "crop_vi": crop_vi,
                "crop_en": crop_en,
                "symptoms": block,
                "pathogen": "Nấm hoặc vi khuẩn gây hại",
                "causes": "Thời tiết ẩm ướt kéo dài.",
                "bio_treatment": "Sử dụng chế phẩm sinh học Trichoderma hoặc Bacillus.",
                "chemical_note": "Chỉ dùng khi cần thiết và cách ly an toàn.",
                "prevention": "Vệ sinh đồng ruộng, tỉa cành thông thoáng.",
                "aliases": [name_part],
            })
    return entries


def build_samples(kb=None, rng=None):
    if rng is None:
        rng = random.Random(SEED)
    if not kb:
        kb = extract_entries_from_markdown()
    diseases = [e for e in kb if e["key"] != "healthy"]
    healthy = next((e for e in kb if e["key"] == "healthy"), None)
    samples = []
    for e in diseases:
        others = [o for o in diseases if o["key"] != e["key"]]
        for focus, tpls in TEMPLATES.items():
            for tpl in tpls:
                ctx = [e, rng.choice(others)]
                rng.shuffle(ctx)
                samples.append({
                    "context": format_context([entry_to_text(c) for c in ctx]),
                    "question": fill(tpl, e),
                    "answer": entry_to_answer(e, focus),
                })
        # Mẫu âm: ngữ cảnh không liên quan -> phải từ chối
        for tpl in (TEMPLATES["full"][1], TEMPLATES["treatment"][0]):
            ctx = rng.sample(others, 2)
            samples.append({
                "context": format_context([entry_to_text(c) for c in ctx]),
                "question": fill(tpl, e),
                "answer": REFUSAL.format(name=e["name_vi"].lower(), crop=e["crop_vi"].lower()),
            })
    if healthy:
        for q in ("Mô hình YOLOv8 phát hiện lá khỏe mạnh (healthy). Tôi cần làm gì để giữ cây khỏe?",
                  "Lá cây của tôi không có bệnh, có cần phun thuốc phòng không?"):
            samples.append({
                "context": format_context([entry_to_text(healthy)]),
                "question": q,
                "answer": entry_to_answer(healthy, "full"),
            })
    extra = TRAIN_DATA_DIR / "llm_extra.jsonl"
    if extra.exists():
        samples += [json.loads(l) for l in extra.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(samples)
    return samples


def make_tokenize_fn(tok, max_length):
    def tokenize(sample):
        msgs = build_messages(sample["context"], sample["question"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": sample["answer"]}], tokenize=False)
        if not full.startswith(prompt):
            full = prompt + sample["answer"] + tok.eos_token
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = [-100] * n_prompt + ids[n_prompt:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels[: len(ids)]}
    return tokenize


def train_custom_llm(base_model: str = LLM_BASE,
                     adapter_dir: Path | str = LLM_ADAPTER_DIR,
                     merged_dir: Path | str = LLM_MERGED_DIR,
                     epochs: float = 3.0,
                     lr: float = 1e-4,
                     max_length: int = 1536,
                     batch: int = 2,
                     grad_accum: int = 4) -> None:
    """Fine-tune mô hình LLM với LoRA và merge weights sang định dạng FP16 độc lập."""
    rng = random.Random(SEED)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samples = build_samples(load_kb(), rng)
    ds = Dataset.from_list(samples).map(make_tokenize_fn(tok, max_length),
                                        remove_columns=["context", "question", "answer"])
    ds = ds.filter(lambda x: any(l != -100 for l in x["labels"]))
    print(f"Số mẫu huấn luyện: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, device_map="auto" if cuda else None)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    chk_dir = Path(LLM_CHECKPOINT_DIR)
    chk_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(chk_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        logging_steps=5,
        save_strategy="no",
        fp16=cuda and not bf16,
        bf16=bf16,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                             pad_to_multiple_of=8),
    )
    trainer.train()

    out_adapter = Path(adapter_dir)
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_adapter))
    tok.save_pretrained(str(out_adapter))
    print(f"Đã lưu adapter LoRA: {out_adapter}")

    del trainer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16 if cuda else torch.float32,
        device_map="auto" if cuda else None)
    merged = PeftModel.from_pretrained(base, str(out_adapter)).merge_and_unload()
    out_merged = Path(merged_dir)
    out_merged.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_merged), safe_serialization=True)
    tok.save_pretrained(str(out_merged))
    print(f"Đã lưu mô hình LLM tự train (đã gộp LoRA) tại: {out_merged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=LLM_BASE)
    ap.add_argument("--adapter-dir", default=str(LLM_ADAPTER_DIR))
    ap.add_argument("--merged-dir", default=str(LLM_MERGED_DIR))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    train_custom_llm(
        base_model=args.base,
        adapter_dir=args.adapter_dir,
        merged_dir=args.merged_dir,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        batch=args.batch,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()
