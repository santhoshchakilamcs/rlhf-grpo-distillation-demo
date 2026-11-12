"""Model-level tests on the smallest model (sshleifer/tiny-gpt2, ~15M).

These verify shapes and mechanics only; the real training runs use
distilgpt2/gpt2 and are covered by the stage scripts.
"""

import torch
from transformers import AutoModelForSequenceClassification

from rlhf_demo.common import (
    STUDENT_MODEL,
    freeze_all_but_last_blocks,
    load_causal_lm,
    load_tokenizer,
)
from rlhf_demo.policy import batch_logprobs, generate_completions
from rlhf_demo.reward_model import RewardModel


def test_reward_model_forward_shape():
    tok = load_tokenizer(STUDENT_MODEL)
    tok.padding_side = "right"
    model = RewardModel(STUDENT_MODEL)
    texts = ["Hello world", "How are you doing today?"]
    enc = tok(texts, padding=True, return_tensors="pt")
    scores = model(enc["input_ids"], enc["attention_mask"])
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_policy_generates_text():
    tok = load_tokenizer(STUDENT_MODEL)
    model = load_causal_lm(STUDENT_MODEL)
    _, completions = generate_completions(
        model, tok, ["Question: What is 2 + 2?\nAnswer:"], max_new_tokens=4, temperature=0.7
    )
    assert len(completions) == 1
    assert isinstance(completions[0], str)


def test_batch_logprobs_shapes_and_mask():
    tok = load_tokenizer(STUDENT_MODEL)
    model = load_causal_lm(STUDENT_MODEL)
    prompts = ["Hello", "The capital of France is"]
    full_texts, _ = generate_completions(model, tok, prompts, max_new_tokens=4, temperature=0.7)
    prompt_lens = [len(tok(p)["input_ids"]) for p in prompts]
    logp, mask = batch_logprobs(model, tok, full_texts, prompt_lens)
    assert logp.shape == mask.shape
    assert mask.dtype == torch.bool
    # Every sequence has at least one real generated token; padded slots masked.
    assert mask.sum(dim=1).min().item() >= 1
    assert torch.isfinite(logp[mask]).all()


def test_freeze_all_but_last_blocks():
    model = load_causal_lm(STUDENT_MODEL)
    trainable = freeze_all_but_last_blocks(model, n_blocks=1)
    assert len(trainable) > 0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert 0 < n_trainable < n_total


def test_sequence_classifier_logits_shape():
    tok = load_tokenizer(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL, num_labels=2, dtype=torch.float32
    )
    model.config.pad_token_id = tok.pad_token_id
    enc = tok(["I loved it", "I hated it"], padding=True, return_tensors="pt")
    logits = model(**enc).logits
    assert logits.shape == (2, 2)
