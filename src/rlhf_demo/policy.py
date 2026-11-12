"""Shared policy helpers for the PPO and GRPO stages.

Generation and per-token log-probability computation for a causal LM on CPU.
Kept separate so both RL stages use identical, tested code.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, max_new_tokens=32, temperature=0.8, top_p=0.95):
    """Sample one completion per prompt. Returns (full_texts, completion_texts)."""
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    completions = []
    for i, plen in enumerate(prompt_lens):
        completions.append(tokenizer.decode(out[i][plen:], skip_special_tokens=True))
    tokenizer.padding_side = "right"
    return full_texts, completions


@torch.no_grad()
def greedy_completions(model, tokenizer, prompts, max_new_tokens=32):
    """Greedy decode, used for before/after evals (deterministic).

    Forces eval mode: a model left in train mode after a policy update
    has dropout active, which makes greedy decoding non-deterministic
    (the same checkpoint scored 0.57 and 0.93 on two train-mode runs).
    Every reported number must be an eval-mode number.
    """
    model.eval()
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    completions = []
    for i, (full, plen) in enumerate(zip(full_texts, prompt_lens)):
        completions.append(tokenizer.decode(out[i][plen:], skip_special_tokens=True))
    tokenizer.padding_side = "right"
    return full_texts, completions


def sequence_logprobs(model, tokenizer, full_texts, prompt_lens):
    """Per-token log-probs of the generated (non-prompt) part of each sequence.

    Returns a list of 1-D tensors, one per sequence. Gradient flows when the
    model is in train mode and inputs require grad.
    """
    enc = tokenizer(full_texts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_logp = torch.gather(logp, 2, target.unsqueeze(-1)).squeeze(-1)
    # token_logp[i, t] is log p of input_ids[i, t+1]; generated tokens start
    # at position prompt_len (the first generated token is at index prompt_len).
    out = []
    for i, plen in enumerate(prompt_lens):
        start = plen - 1  # index in token_logp of the first generated token
        n_gen = input_ids.size(1) - plen
        n_gen = min(n_gen, token_logp.size(1) - start)
        seq_lp = token_logp[i, start : start + n_gen]
        # mask out padding positions (attention 0 beyond the true length)
        true_len = int(attention_mask[i].sum().item())
        n_real = true_len - plen
        seq_lp = seq_lp[: max(n_real, 0)]
        out.append(seq_lp)
    return out


def batch_logprobs(model, tokenizer, full_texts, prompt_lens):
    """Stacked per-token log-probs, padded with zeros, plus a mask.

    Returns (logp_padded [B, T], mask [B, T] bool) where mask marks real
    generated tokens.
    """
    seqs = sequence_logprobs(model, tokenizer, full_texts, prompt_lens)
    maxlen = max(s.size(0) for s in seqs) if seqs else 0
    logp = torch.zeros(len(seqs), maxlen)
    mask = torch.zeros(len(seqs), maxlen, dtype=torch.bool)
    for i, s in enumerate(seqs):
        logp[i, : s.size(0)] = s
        mask[i, : s.size(0)] = True
    return logp, mask


def clipped_policy_update(
    policy,
    tokenizer,
    full_texts,
    prompt_lens,
    old_logp,
    mask,
    advantages,
    opt,
    clip_eps=0.2,
    kl_beta=0.0,
    kl_per_seq=None,
):
    """One clipped-surrogate policy-gradient update (PPO/GRPO shared core).

    ``advantages`` is a per-sequence tensor. Optionally adds a KL penalty
    ``kl_beta * mean(kl_per_seq)``. Returns the loss value as a float.
    The caller owns the optimizer so Adam state survives across epochs.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    policy.train()
    new_logp, _ = batch_logprobs(policy, tokenizer, full_texts, prompt_lens)
    ratio = torch.exp((new_logp - old_logp) * mask)
    surr1 = ratio * advantages.unsqueeze(1)
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.unsqueeze(1)
    per_token = torch.minimum(surr1, surr2)
    loss = -(per_token * mask).sum() / mask.sum()
    if kl_per_seq is not None and kl_beta:
        loss = loss + kl_beta * kl_per_seq.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    return float(loss.item())
