"""Part 1b: minimal from-scratch PPO fine-tuning of the policy.

This is NOT TRL. A full TRL PPO run needs GPU-scale generation; on this
CPU-only box we implement the same core algorithm directly:
- rollouts sampled from the policy
- reward = reward-model score minus a KL penalty vs a frozen reference
- scalar baseline (running mean) for the advantage
- clipped-surrogate PPO update (clip 0.2), a few epochs per batch

Only the last two transformer blocks of distilgpt2 are trainable (see
common.freeze_all_but_last_blocks). The reference model is loaded only to
score ref log-probs for a batch, then freed, so peak memory stays low:
the policy, reference and reward model are never all resident at once.
"""

from __future__ import annotations

import json

import torch

from .common import (
    CHECKPOINTS_DIR,
    POLICY_MODEL,
    RESULTS_DIR,
    freeze_all_but_last_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
)
from .data import load_hh_subset
from .policy import (
    batch_logprobs,
    clipped_policy_update,
    generate_completions,
    greedy_completions,
)
from .reward_model import load_reward_model, reward_scores

KL_BETA = 0.1
CLIP_EPS = 0.2
PPO_EPOCHS = 2
ASSISTANT_MARKER = "\n\nAssistant:"
# Anti-degeneracy shaping: a weak reward model can be gamed by repeating a
# high-scoring phrase (first run: the policy converged on repeating
# "Assistant:"). Penalise completions whose bigrams are mostly repeats.
REP_PENALTY_WEIGHT = 0.5


def repetition_penalty(text: str) -> float:
    """Fraction of bigrams that are duplicates (0 = no repeats, ~1 = loop).

    "Assistant: Assistant: ..." scores ~1.0; ordinary prose scores ~0.1-0.3.
    """
    words = text.split()
    if len(words) < 4:
        return 0.0
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def eval_policy_reward(policy, tokenizer, reward_model, prompts, max_new_tokens=32):
    """Mean reward-model score of greedy completions for fixed prompts.

    Reports both the raw reward-model score and the repetition-penalised
    score, so reward hacking is visible instead of hidden.
    """
    _, completions = greedy_completions(policy, tokenizer, prompts, max_new_tokens=max_new_tokens)
    texts = [p + c for p, c in zip(prompts, completions)]
    scores = reward_scores(reward_model, tokenizer, texts)
    penalties = [repetition_penalty(c) for c in completions]
    penalised = [s - REP_PENALTY_WEIGHT * p for s, p in zip(scores, penalties)]
    lengths = [len(c.split()) for c in completions]
    return {
        "mean_reward": sum(scores) / len(scores),
        "mean_penalty": sum(penalties) / len(penalties),
        "mean_reward_penalised": sum(penalised) / len(penalised),
        "mean_completion_words": sum(lengths) / len(lengths),
        "completions": completions,
        "scores": scores,
    }


def run_ppo(
    n_prompts: int = 32,
    n_eval_prompts: int = 24,
    iterations: int = 8,
    batch_prompts: int = 8,
    max_new_tokens: int = 32,
    lr: float = 1e-5,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    torch.set_num_threads(4)
    tok = load_tokenizer(POLICY_MODEL)
    tok.padding_side = "right"

    # Prompts: held-out HH-RLHF human turns, disjoint from the reward
    # model's train (pairs 0-320) and eval (320-400) pairs.
    pairs = load_hh_subset(400, 60, seed=seed)
    train_prompts = [p["prompt"] + ASSISTANT_MARKER for p in pairs[400:400 + n_prompts]]
    eval_prompts = [p["prompt"] + ASSISTANT_MARKER for p in pairs[400 + n_prompts:400 + n_prompts + n_eval_prompts]]

    reward_model, _ = load_reward_model()

    policy = load_causal_lm(POLICY_MODEL)
    trainable = freeze_all_but_last_blocks(policy, n_blocks=2)
    print(f"PPO policy: {len(trainable)} trainable tensors (last 2 blocks)")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)

    before = eval_policy_reward(policy, tok, reward_model, eval_prompts, max_new_tokens)
    print(f"  before PPO: mean reward {before['mean_reward']:.3f}")

    baseline = before["mean_reward"]
    kl_history: list[float] = []
    reward_history: list[float] = []

    for it in range(iterations):
        # Cycle through the prompt list so `iterations` really runs that many
        # batches (32 prompts / 8 per batch = 4 unique batches; wrap around).
        batch = [
            train_prompts[(it * batch_prompts + j) % len(train_prompts)]
            for j in range(batch_prompts)
        ]
        # 1. Rollouts with the current policy.
        policy.eval()
        full_texts, completions = generate_completions(
            policy, tok, batch, max_new_tokens=max_new_tokens, temperature=0.8
        )
        prompt_lens = [len(tok(p)["input_ids"]) for p in batch]
        # 2. Rewards from the reward model, minus the repetition penalty.
        scores = reward_scores(reward_model, tok, full_texts)
        penalties = [repetition_penalty(c) for c in completions]
        penalised = [
            s - REP_PENALTY_WEIGHT * p for s, p in zip(scores, penalties)
        ]
        # 3. Old log-probs (policy) and ref log-probs (reference, loaded briefly).
        with torch.no_grad():
            old_logp, mask = batch_logprobs(policy, tok, full_texts, prompt_lens)
        ref_model = load_causal_lm(POLICY_MODEL)
        ref_model.eval()
        with torch.no_grad():
            ref_logp, _ = batch_logprobs(ref_model, tok, full_texts, prompt_lens)
        del ref_model
        # 4. Per-sequence KL(π_policy || π_ref), estimated as the summed
        # per-token log-ratio log π_policy - log π_ref over generated tokens.
        kl_per_seq = ((old_logp - ref_logp) * mask).sum(dim=1)
        rewards = (
            torch.tensor(penalised, dtype=torch.float32) - KL_BETA * kl_per_seq
        )
        baseline = 0.9 * baseline + 0.1 * rewards.mean().item()
        adv = rewards - baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)
        kl_history.append(float(kl_per_seq.mean()))
        reward_history.append(float(torch.tensor(penalised).mean()))
        print(
            f"  iter {it + 1}/{iterations}: mean penalised reward {torch.tensor(penalised).mean():.3f} "
            f"(raw {torch.tensor(scores).mean():.3f}, rep penalty {sum(penalties) / len(penalties):.3f}), "
            f"mean KL {kl_per_seq.mean():.3f}"
        )
        # 5. Clipped-surrogate update (recompute log-probs with gradients).
        for _ in range(PPO_EPOCHS):
            clipped_policy_update(
                policy, tok, full_texts, prompt_lens, old_logp, mask, adv, opt, CLIP_EPS
            )
        del old_logp, ref_logp

    after = eval_policy_reward(policy, tok, reward_model, eval_prompts, max_new_tokens)
    print(f"  after PPO: mean reward {after['mean_reward']:.3f}")

    out_dir = CHECKPOINTS_DIR / "ppo_policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / "pytorch_model.bin")

    results = {
        "part": "ppo",
        "iterations": iterations,
        "batch_prompts": batch_prompts,
        "kl_beta": KL_BETA,
        "clip_eps": CLIP_EPS,
        "rep_penalty_weight": REP_PENALTY_WEIGHT,
        "eval_prompts": n_eval_prompts,
        "mean_reward_before": round(before["mean_reward"], 4),
        "mean_reward_after": round(after["mean_reward"], 4),
        "mean_reward_penalised_before": round(before["mean_reward_penalised"], 4),
        "mean_reward_penalised_after": round(after["mean_reward_penalised"], 4),
        "mean_rep_penalty_before": round(before["mean_penalty"], 4),
        "mean_rep_penalty_after": round(after["mean_penalty"], 4),
        "mean_completion_words_before": round(before["mean_completion_words"], 2),
        "mean_completion_words_after": round(after["mean_completion_words"], 2),
        "mean_kl_per_token_during_training": round(sum(kl_history) / len(kl_history), 4) if kl_history else None,
        "sample_completion_before": before["completions"][0],
        "sample_completion_after": after["completions"][0],
        "sample_prompt": eval_prompts[0],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ppo.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(run_ppo(), indent=2))
