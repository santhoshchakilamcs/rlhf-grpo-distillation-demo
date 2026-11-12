"""Shared helpers: seeding, device, and tiny-model loading.

Everything runs on CPU with small open models. No GPU is used anywhere.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Small open models only. distilgpt2 (82M) is the workhorse; gpt2 (124M) is
# the distillation teacher. sshleifer/tiny-gpt2 is used only by the fast
# test suite (its hidden size is 2, so it cannot learn the distillation
# task); the distillation student is a from-config 4-layer GPT-2 built in
# distill.py.
POLICY_MODEL = "distilgpt2"
REWARD_MODEL = "distilgpt2"
TEACHER_MODEL = "gpt2"
STUDENT_MODEL = "sshleifer/tiny-gpt2"

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_device() -> torch.device:
    return torch.device("cpu")


def load_tokenizer(model_name: str = POLICY_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(model_name: str = POLICY_MODEL) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    return count_params(model) * 4 / 1e6  # fp32


def freeze_all_but_last_blocks(model: torch.nn.Module, n_blocks: int = 2) -> list[str]:
    """Freeze everything except the last ``n_blocks`` transformer blocks.

    Returns the names of the trainable parameters. The lm_head is tied to
    the input embeddings in GPT-2, so it stays frozen too: training the last
    blocks is enough to shift the output distribution, and it keeps the
    AdamW state small enough for a CPU-only box.
    """
    for p in model.parameters():
        p.requires_grad = False
    # GPT-2 style: GPT2LMHeadModel / GPT2ForSequenceClassification keep the
    # blocks at model.transformer.h; a bare GPT2Model keeps them at model.h.
    if hasattr(model, "transformer"):
        blocks = model.transformer.h
    else:
        blocks = model.h
    trainable: list[str] = []
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)
    return trainable


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
