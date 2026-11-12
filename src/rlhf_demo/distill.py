"""Part 3: knowledge distillation on SST-2 sentiment classification.

Teacher: gpt2 (124M) fine-tuned on SST-2 (head + last two blocks).
Student: a small GPT-2 built from a config (4 layers, hidden 384, ~27M
params), randomly initialised and trained from scratch two ways:
  (a) distillation: KL to the teacher's soft labels (T=2) + cross-entropy
  (b) baseline: cross-entropy only
Report: accuracy of teacher / distilled student / CE-only student, plus
parameter counts, fp32 size, and mean per-example inference latency on CPU.

Why a from-config student: the only tiny GPT-2 on the Hub,
sshleifer/tiny-gpt2, has hidden size 2 and cannot learn the task at all,
so a small from-scratch student is the honest demo-size choice.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
)

from .common import (
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    TEACHER_MODEL,
    count_params,
    model_size_mb,
    set_seed,
)
from .data import load_sst2_subset

TEMPERATURE = 2.0
ALPHA_CE = 0.3  # weight on the true-label CE term; 1-ALPHA_CE on the KL term
STUDENT_NAME = "gpt2-4L-384d (from-scratch)"


def student_config() -> GPT2Config:
    cfg = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=384,
        n_layer=4,
        n_head=6,
        pad_token_id=50256,
    )
    cfg.num_labels = 2
    return cfg


class SentimentDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int = 64):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


def _collate(batch, pad_id: int = 50256):
    """Pad variable-length sequences. Pad with the GPT-2 pad/eos id (50256),
    not 0: GPT2ForSequenceClassification pools the last non-pad token, so
    padding with a real token id would make it pool a padding position."""
    maxlen = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    att = torch.zeros(len(batch), maxlen, dtype=torch.long)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        ids[i, : b["input_ids"].size(0)] = b["input_ids"]
        att[i, : b["attention_mask"].size(0)] = b["attention_mask"]
    return ids, att, labels


def _trainable_last_blocks(model, n_blocks: int = 2):
    """Freeze all but the last ``n_blocks`` transformer blocks + score head.

    GPT2ForSequenceClassification keeps the base model at ``model.transformer``
    and the classification head at ``model.score``.
    """
    for p in model.parameters():
        p.requires_grad = False
    for block in model.transformer.h[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.score.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def accuracy(model, rows, tokenizer, batch_size: int = 16, max_len: int = 64) -> float:
    model.eval()
    loader = DataLoader(SentimentDataset(rows, tokenizer, max_len), batch_size=batch_size, collate_fn=_collate)
    correct, total = 0, 0
    for ids, att, labels in loader:
        logits = model(input_ids=ids, attention_mask=att).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def mean_latency_ms(model, rows, tokenizer, n: int = 100, max_len: int = 64) -> float:
    """Mean per-example forward-pass time, batch size 1, CPU."""
    model.eval()
    subset = rows[:n]
    times: list[float] = []
    for r in subset:
        enc = tokenizer(r["text"], truncation=True, max_length=max_len, return_tensors="pt")
        start = time.perf_counter()
        model(**enc)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times) * 1000


def _train_classifier(model, train_rows, tokenizer, epochs=2, batch_size=16, lr=2e-5, lr_head=1e-3, max_len=64, train_all=False):
    """Train a sequence classifier. The classification head starts randomly
    initialised, so it gets its own (higher) learning rate; at lr 2e-5 the
    head barely moves in 2 epochs and the model stays at chance (first
    attempt: teacher accuracy 0.520)."""
    if train_all:
        for p in model.parameters():
            p.requires_grad = True
        params = list(model.parameters())
    else:
        params = _trainable_last_blocks(model, n_blocks=2)
    head_params = [p for n, p in model.named_parameters() if "score" in n and p.requires_grad]
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in params if id(p) not in head_ids]
    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    opt = torch.optim.AdamW(groups)
    loader = DataLoader(
        SentimentDataset(train_rows, tokenizer, max_len), batch_size=batch_size, shuffle=True, collate_fn=_collate
    )
    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for ids, att, labels in loader:
            logits = model(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"    classifier epoch {epoch + 1}/{epochs}: mean loss {total_loss / n_batches:.4f}")
    return model


def _train_student_distill(teacher, student, train_rows, tokenizer, epochs=2, batch_size=16, lr=1e-4, max_len=64):
    teacher.eval()
    for p in student.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    loader = DataLoader(
        SentimentDataset(train_rows, tokenizer, max_len), batch_size=batch_size, shuffle=True, collate_fn=_collate
    )
    student.train()
    for _ in range(epochs):
        for ids, att, labels in loader:
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=att).logits
            s_logits = student(input_ids=ids, attention_mask=att).logits
            ce = F.cross_entropy(s_logits, labels)
            kl = F.kl_div(
                F.log_softmax(s_logits / TEMPERATURE, dim=-1),
                F.softmax(t_logits / TEMPERATURE, dim=-1),
                reduction="batchmean",
            ) * (TEMPERATURE ** 2)
            loss = ALPHA_CE * ce + (1 - ALPHA_CE) * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
    return student


def run_distillation(
    n_train: int = 500,
    n_eval: int = 200,
    epochs: int = 2,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    torch.set_num_threads(4)
    train_rows, eval_rows = load_sst2_subset(n_train, n_eval, seed=seed)
    source = train_rows[0]["source"]

    tok_t = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tok_t.pad_token is None:
        tok_t.pad_token = tok_t.eos_token
    tok_s = tok_t  # same GPT-2 vocabulary for teacher and student

    # Teacher: fine-tune gpt2 on SST-2 (full model; the head is random, so
    # it trains at a higher LR — see _train_classifier).
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, num_labels=2, dtype=torch.float32
    )
    teacher.config.pad_token_id = tok_t.pad_token_id
    print("training teacher (gpt2) ...")
    _train_classifier(teacher, train_rows, tok_t, epochs=3, lr=2e-5, train_all=True)
    teacher_acc = accuracy(teacher, eval_rows, tok_t)
    print(f"  teacher accuracy: {teacher_acc:.3f}")

    # Student, distilled from the teacher (from-scratch small GPT-2).
    student = AutoModelForSequenceClassification.from_config(student_config())
    student.config.pad_token_id = tok_s.pad_token_id
    print("training student with distillation ...")
    _train_student_distill(teacher, student, train_rows, tok_s, epochs=epochs + 1)
    student_acc = accuracy(student, eval_rows, tok_s)
    print(f"  distilled student accuracy: {student_acc:.3f}")

    # Student baseline: same architecture, cross-entropy only.
    student_ce = AutoModelForSequenceClassification.from_config(student_config())
    student_ce.config.pad_token_id = tok_s.pad_token_id
    print("training student baseline (cross-entropy only) ...")
    _train_classifier(student_ce, train_rows, tok_s, epochs=epochs + 1, lr=1e-4, train_all=True)
    student_ce_acc = accuracy(student_ce, eval_rows, tok_s)
    print(f"  CE-only student accuracy: {student_ce_acc:.3f}")

    teacher_lat = mean_latency_ms(teacher, eval_rows, tok_t)
    student_lat = mean_latency_ms(student, eval_rows, tok_s)

    out_dir = CHECKPOINTS_DIR / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), out_dir / "teacher.bin")
    torch.save(student.state_dict(), out_dir / "student_distilled.bin")
    torch.save(student_ce.state_dict(), out_dir / "student_ce_only.bin")

    results = {
        "part": "distillation",
        "task": "SST-2 sentiment classification (positive/negative)",
        "data_source": source,
        "n_train": n_train,
        "n_eval": n_eval,
        # The teacher runs 3 epochs (hardcoded above); both students run
        # epochs + 1. Record the actual numbers, not the base parameter.
        "epochs": epochs + 1,
        "teacher_epochs": 3,
        "temperature": TEMPERATURE,
        "alpha_ce": ALPHA_CE,
        "teacher": TEACHER_MODEL,
        "student": STUDENT_NAME,
        "teacher_params": count_params(teacher),
        "student_params": count_params(student),
        "teacher_size_mb": round(model_size_mb(teacher), 1),
        "student_size_mb": round(model_size_mb(student), 1),
        "compression_ratio": round(count_params(teacher) / count_params(student), 2),
        "teacher_accuracy": round(teacher_acc, 4),
        "student_distilled_accuracy": round(student_acc, 4),
        "student_ce_only_accuracy": round(student_ce_acc, 4),
        "teacher_latency_ms_per_example": round(teacher_lat, 2),
        "student_latency_ms_per_example": round(student_lat, 2),
        "latency_speedup": round(teacher_lat / student_lat, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "distillation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(run_distillation(), indent=2))
