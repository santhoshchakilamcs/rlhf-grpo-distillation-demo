"""Part 1a: reward model trained on human preferences (Bradley-Terry).

Architecture: distilgpt2 backbone with the last two transformer blocks
trainable, plus a scalar reward head on the last-token hidden state.
Loss: -log sigmoid(r_chosen - r_rejected).

The trained reward model is saved to checkpoints/reward_model/ and reused
by the PPO stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from .common import (
    CHECKPOINTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    REWARD_MODEL,
    freeze_all_but_last_blocks,
    load_tokenizer,
    set_seed,
)
from .data import load_hh_subset


class RewardModel(nn.Module):
    def __init__(self, model_name: str = REWARD_MODEL):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_idx = attention_mask.sum(dim=1) - 1
        last_hidden = out.last_hidden_state[torch.arange(input_ids.size(0)), last_idx]
        return self.head(last_hidden).squeeze(-1)


class PreferenceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len: int = 128):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def _encode(self, prompt: str, response: str):
        """Encode prompt + response, keeping the response fully visible.

        The chosen and rejected texts share the whole conversation history
        and differ only in the final assistant response. Right-truncating
        the concatenated text (the first attempt) cuts the response off
        for long conversations, so the model sees identical inputs and
        cannot learn anything. Instead the shared prompt is truncated
        from the left and the response is kept (up to 96 tokens).
        """
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = self.tok("\n\nAssistant: " + response, add_special_tokens=False)["input_ids"]
        r_ids = r_ids[-96:]
        budget = self.max_len - len(r_ids)
        p_ids = p_ids[-max(budget, 0):]
        ids = p_ids + r_ids
        return torch.tensor(ids, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)

    def __getitem__(self, i):
        p = self.pairs[i]
        c_ids, c_mask = self._encode(p["prompt"], p["chosen"])
        r_ids, r_mask = self._encode(p["prompt"], p["rejected"])
        return {
            "chosen_ids": c_ids,
            "chosen_mask": c_mask,
            "rejected_ids": r_ids,
            "rejected_mask": r_mask,
        }


def _collate(batch):
    def pad(key, mask_key):
        seqs = [b[key] for b in batch]
        masks = [b[mask_key] for b in batch]
        maxlen = max(s.size(0) for s in seqs)
        ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        att = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids[i, : s.size(0)] = s
            att[i, : m.size(0)] = m
        return ids, att

    c_ids, c_mask = pad("chosen_ids", "chosen_mask")
    r_ids, r_mask = pad("rejected_ids", "rejected_mask")
    return c_ids, c_mask, r_ids, r_mask


@torch.no_grad()
def reward_scores(model, tokenizer, texts, max_len: int = 128, batch_size: int = 8) -> list[float]:
    """Score prompt+response texts with the reward model.

    Left-truncates: the response (the scored part) sits at the end of the
    text and must not be cut off.
    """
    model.eval()
    prev_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
        )
        s = model(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.tolist())
    tokenizer.truncation_side = prev_side
    return scores


def train_reward_model(
    n_train: int = 320,
    n_eval: int = 80,
    epochs: int = 4,
    batch_size: int = 8,
    lr_blocks: float = 2e-5,
    lr_head: float = 1e-3,
    max_len: int = 160,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    torch.set_num_threads(4)
    pairs = load_hh_subset(n_train, n_eval, seed=seed)
    train_pairs, eval_pairs = pairs[:n_train], pairs[n_train:]
    source = pairs[0]["source"]
    tok = load_tokenizer(REWARD_MODEL)
    tok.padding_side = "right"
    model = RewardModel(REWARD_MODEL)
    trainable = freeze_all_but_last_blocks(model.backbone, n_blocks=2)
    for p in model.head.parameters():
        p.requires_grad = True
    print(f"reward model: {len(trainable)} trainable tensors (last 2 blocks + head)")

    # The head is randomly initialised, so it needs a much larger step than
    # the pretrained blocks; a single 1e-5 rate leaves it a random projection
    # (first attempt: held-out accuracy 0.33, worse than chance).
    block_params = [p for name, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": block_params, "lr": lr_blocks},
            {"params": head_params, "lr": lr_head},
        ]
    )
    loader = DataLoader(
        PreferenceDataset(train_pairs, tok, max_len=max_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    model.train()
    for epoch in range(epochs):
        total_loss, steps = 0.0, 0
        for c_ids, c_mask, r_ids, r_mask in loader:
            r_c = model(c_ids, c_mask)
            r_r = model(r_ids, r_mask)
            loss = -torch.nn.functional.logsigmoid(r_c - r_r).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            total_loss += loss.item()
            steps += 1
        print(f"  epoch {epoch + 1}/{epochs} train BT loss {total_loss / max(steps, 1):.4f}")

    # Held-out accuracy: how often the chosen response scores above rejected.
    chosen_texts = [p["prompt"] + "\n\nAssistant: " + p["chosen"] for p in eval_pairs]
    rejected_texts = [p["prompt"] + "\n\nAssistant: " + p["rejected"] for p in eval_pairs]
    r_c = reward_scores(model, tok, chosen_texts, max_len=max_len)
    r_r = reward_scores(model, tok, rejected_texts, max_len=max_len)
    correct = sum(1 for a, b in zip(r_c, r_r) if a > b)
    acc = correct / len(eval_pairs)
    gap = sum(a - b for a, b in zip(r_c, r_r)) / len(eval_pairs)
    print(f"  held-out preference accuracy: {acc:.3f} ({correct}/{len(eval_pairs)})")
    print(f"  mean chosen-rejected reward gap: {gap:.3f}")
    if acc < 0.5:
        print("  WARNING: held-out accuracy below chance; the reward signal is not usable for PPO.")
    # Train-set accuracy on a subsample: confirms the signal is visible at all.
    train_sample = train_pairs[:60]
    tr_c = reward_scores(
        model, tok,
        [p["prompt"] + "\n\nAssistant: " + p["chosen"] for p in train_sample],
        max_len=max_len,
    )
    tr_r = reward_scores(
        model, tok,
        [p["prompt"] + "\n\nAssistant: " + p["rejected"] for p in train_sample],
        max_len=max_len,
    )
    train_acc = sum(1 for a, b in zip(tr_c, tr_r) if a > b) / len(train_sample)
    print(f"  train-subsample preference accuracy: {train_acc:.3f} ({len(train_sample)} pairs)")

    out_dir = CHECKPOINTS_DIR / "reward_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": REWARD_MODEL, "head": "linear-scalar"}, f, indent=2)

    results = {
        "part": "reward_model",
        "data_source": source,
        "n_train": n_train,
        "n_eval": n_eval,
        "epochs": epochs,
        "lr_blocks": lr_blocks,
        "lr_head": lr_head,
        "max_len": max_len,
        "held_out_accuracy": round(acc, 4),
        "held_out_correct": correct,
        "mean_reward_gap": round(gap, 4),
        "train_subsample_accuracy": round(train_acc, 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "reward_model.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def load_reward_model(checkpoint_dir: Path | None = None) -> tuple[RewardModel, object]:
    tok = load_tokenizer(REWARD_MODEL)
    model = RewardModel(REWARD_MODEL)
    ckpt = (checkpoint_dir or CHECKPOINTS_DIR / "reward_model") / "pytorch_model.bin"
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return model, tok


if __name__ == "__main__":
    print(json.dumps(train_reward_model(), indent=2))
