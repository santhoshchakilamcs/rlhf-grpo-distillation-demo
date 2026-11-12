"""Smoke tests: one real update step of each RL loop on the tiny model.

These run the actual training code paths (generate -> log-probs -> clipped
update) with 2 prompts, asserting the loss is finite and parameters move.
"""

import torch

from rlhf_demo.common import (
    STUDENT_MODEL,
    freeze_all_but_last_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
)
from rlhf_demo.policy import (
    batch_logprobs,
    clipped_policy_update,
    generate_completions,
    greedy_completions,
)
from rlhf_demo.reward_model import RewardModel


def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def test_one_clipped_policy_update_moves_params():
    set_seed(42)
    torch.set_num_threads(2)
    tok = load_tokenizer(STUDENT_MODEL)
    model = load_causal_lm(STUDENT_MODEL)
    freeze_all_but_last_blocks(model, n_blocks=1)
    before = [p.detach().clone() for p in _trainable_params(model)]

    prompts = ["Question: What is 3 + 4?\nAnswer:", "Question: What is 9 - 2?\nAnswer:"]
    model.eval()
    full_texts, _ = generate_completions(model, tok, prompts, max_new_tokens=6, temperature=0.9)
    prompt_lens = [len(tok(p)["input_ids"]) for p in prompts]
    with torch.no_grad():
        old_logp, mask = batch_logprobs(model, tok, full_texts, prompt_lens)
    advantages = torch.tensor([1.0, -1.0])
    opt = torch.optim.AdamW(_trainable_params(model), lr=1e-4)
    loss = clipped_policy_update(
        model, tok, full_texts, prompt_lens, old_logp, mask, advantages, opt, 0.2
    )
    assert loss == loss  # finite (not NaN)
    after = [p.detach().clone() for p in _trainable_params(model)]
    moved = any(not torch.equal(a, b) for a, b in zip(before, after))
    assert moved, "trainable parameters did not change after one update"


def test_one_reward_model_step():
    set_seed(42)
    torch.set_num_threads(2)
    tok = load_tokenizer(STUDENT_MODEL)
    tok.padding_side = "right"
    model = RewardModel(STUDENT_MODEL)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    head_before = [p.detach().clone() for p in model.head.parameters()]
    pairs = [
        {"prompt": "How do I bake bread?", "chosen": "Here is a helpful recipe.", "rejected": "No."},
        {"prompt": "What is 2+2?", "chosen": "2+2 is 4.", "rejected": "I don't care."},
    ]
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    c_texts = [p["prompt"] + "\n\nAssistant: " + p["chosen"] for p in pairs]
    r_texts = [p["prompt"] + "\n\nAssistant: " + p["rejected"] for p in pairs]
    enc_c = tok(c_texts, padding=True, return_tensors="pt")
    enc_r = tok(r_texts, padding=True, return_tensors="pt")
    r_c = model(enc_c["input_ids"], enc_c["attention_mask"])
    r_r = model(enc_r["input_ids"], enc_r["attention_mask"])
    loss = -torch.nn.functional.logsigmoid(r_c - r_r).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    head_after = [p.detach().clone() for p in model.head.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(head_before, head_after))


def test_greedy_completions_forces_eval_mode_and_is_reproducible():
    """Regression test for the eval-mode bug (Nov 12, 2026).

    greedy_completions must force eval mode: a model left in train mode
    after a policy update has dropout active, so the same checkpoint
    scored 0.57 and 0.93 on two train-mode greedy evals. Evals must be
    deterministic and identical across calls.
    """
    set_seed(42)
    torch.set_num_threads(2)
    tok = load_tokenizer(STUDENT_MODEL)
    model = load_causal_lm(STUDENT_MODEL)
    model.train()  # simulate a model left in train mode after an update
    prompts = ["Hello, my name is", "The capital of France is"]
    _, completions_1 = greedy_completions(model, tok, prompts, max_new_tokens=8)
    assert model.training is False
    _, completions_2 = greedy_completions(model, tok, prompts, max_new_tokens=8)
    assert completions_1 == completions_2
