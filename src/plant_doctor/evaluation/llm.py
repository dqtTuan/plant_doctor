"""Đánh giá LLM (sau LoRA) trên tập test held-out (bệnh chưa thấy khi train).

- Chất lượng câu trả lời (mẫu "answer"): ROUGE-L F1, token-F1 so với đáp án mẫu
- Chống ảo giác: precision/recall/F1 của hành vi từ chối, tỉ lệ từ chối nhầm câu có đáp án
Kết quả từng mẫu được lưu predictions.jsonl để soi lỗi.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from statistics import mean

from plant_doctor.evaluation.metrics import is_refusal, refusal_metrics, rouge_l, token_f1


def select_samples(samples: list[dict], max_samples: int | None) -> list[dict]:
    """Giữ tỉ lệ answer/refusal khi cắt bớt mẫu (thứ tự xác định)."""
    if not max_samples or len(samples) <= max_samples:
        return list(samples)
    answers = [s for s in samples if s.get("kind") != "refusal"]
    refusals = [s for s in samples if s.get("kind") == "refusal"]
    n_ref = round(max_samples * len(refusals) / len(samples))
    n_ref = min(len(refusals), max(n_ref, 1 if refusals else 0))
    return answers[: max_samples - n_ref] + refusals[:n_ref]


def score_predictions(samples: Iterable[dict], predictions: Iterable[str]) -> tuple[dict, list[dict]]:
    rows = []
    for s, pred in zip(samples, predictions):
        should = s.get("kind") == "refusal"
        rows.append({
            "question": s["question"], "kind": s.get("kind", "answer"), "reference": s["answer"],
            "prediction": pred, "refused": is_refusal(pred),
            "rouge_l": rouge_l(pred, s["answer"]), "token_f1": token_f1(pred, s["answer"]),
            "should_refuse": should,
        })
    answers = [r for r in rows if not r["should_refuse"]]
    metrics = {
        "n_samples": len(rows),
        "rouge_l": mean(r["rouge_l"] for r in answers) if answers else 0.0,
        "token_f1": mean(r["token_f1"] for r in answers) if answers else 0.0,
        **refusal_metrics([r["should_refuse"] for r in rows], [r["refused"] for r in rows]),
    }
    return metrics, rows


def make_generator(model, tokenizer, max_new_tokens: int = 384) -> Callable[[dict], str]:
    """Hàm sinh câu trả lời greedy dùng đúng prompt như lúc suy luận trong app."""
    import torch

    from plant_doctor.inference.prompts import build_messages

    def generate(sample: dict) -> str:
        msgs = build_messages(sample["context"], sample["question"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.1,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return generate


def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16 if cuda else torch.float32,
                                                 device_map="auto" if cuda else None)
    return model.eval(), tok


def evaluate_llm(generate: Callable[[dict], str], samples: list[dict], max_samples: int | None = None
                 ) -> tuple[dict, list[dict]]:
    from tqdm import tqdm

    chosen = select_samples(samples, max_samples)
    preds = [generate(s) for s in tqdm(chosen, desc="Đánh giá LLM")]
    return score_predictions(chosen, preds)

