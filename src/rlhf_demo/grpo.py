"""Part 2: minimal from-scratch GRPO fine-tuning on a verifiable-reward task.

This is NOT TRL. A full TRL GRPO run needs GPU-scale generation; on this
CPU-only box we implement the core GRPO idea directly (DeepSeekMath-style):
- for each prompt, sample a group of G completions from the policy
- score each with a programmatic, verifiable reward (exact arithmetic)
- group-relative advantage: (r - group mean) / (group std + eps)
- clipped policy-gradient update with a KL penalty vs a frozen reference

Task: short addition/subtraction, prompt ends with "Answer:" so the model
only has to emit the number. Reward tiers are in data.math_reward.
Only the last two transformer blocks of distilgpt2 are trainable.

Pipeline order matters: GRPO alone on the base model does not learn here.
A base distilgpt2 never emits a bare number after "Answer:", so correct
answers are almost never sampled and the group-relative advantage has
nothing to amplify (two attempts: exact match stayed 0.000 after 8 steps
on two-digit and 12 steps on single-digit problems). So we warm-start with
a short SFT pass on the same train problems (loss on the answer tokens
only) to teach the answer format; GRPO then refines *which* number is
emitted. Both the pre-SFT baseline and the post-SFT / pre-GRPO baseline
are reported, so the GRPO delta is isolated.
"""

from __future__ import annotations

import json
import random

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
from .data import make_math_problems, math_exact_match, math_reward
from .policy import (
    batch_logprobs,
    clipped_policy_update,
    generate_completions,
    greedy_completions,
)

GROUP_SIZE = 4
KL_BETA = 0.02
CLIP_EPS = 0.2


@torch.no_grad()
def eval_math(policy, tokenizer, problems, max_new_tokens=24):
    """Greedy-decode eval on held-out problems: mean reward, exact-match, format."""
    prompts = [p["prompt"] for p in problems]
    _, completions = greedy_completions(policy, tokenizer, prompts, max_new_tokens=max_new_tokens)
    rewards = [math_reward(c, p["answer"]) for c, p in zip(completions, problems)]
    exact = [math_exact_match(c, p["answer"]) for c, p in zip(completions, problems)]
    has_number = [len(c.strip()) > 0 and any(ch.isdigit() for ch in c) for c in completions]
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "exact_match_rate": sum(exact) / len(exact),
        "any_number_rate": sum(has_number) / len(has_number),
        "completions": completions,
    }


def sft_warm_start(policy, tokenizer, problems, epochs=3, lr=2e-5, batch_size=16, seed=42):
    """SFT warm-start: teach the policy the answer format (a bare number
    after 'Answer:') before GRPO. Loss is on the answer tokens only; the
    prompt is masked. Without this step GRPO on the base model finds almost
    no correct answers to amplify (see module docstring)."""
    rng = random.Random(seed)
    policy.train()
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    pad_id = tokenizer.pad_token_id
    for epoch in range(epochs):
        order = list(range(len(problems)))
        rng.shuffle(order)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(order), batch_size):
            batch_ids, batch_labels = [], []
            for j in order[i:i + batch_size]:
                p = problems[j]
                prompt_ids = tokenizer(p["prompt"])["input_ids"]
                answer_ids = tokenizer(f" {p['answer']}")["input_ids"]
                batch_ids.append(prompt_ids + answer_ids)
                batch_labels.append([-100] * len(prompt_ids) + answer_ids)
            maxlen = max(len(x) for x in batch_ids)
            input_ids = torch.tensor([x + [pad_id] * (maxlen - len(x)) for x in batch_ids])
            labels = torch.tensor([x + [-100] * (maxlen - len(x)) for x in batch_labels])
            attn = (input_ids != pad_id).long()
            out = policy(input_ids=input_ids, attention_mask=attn, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
            n_batches += 1
        print(f"  SFT epoch {epoch + 1}/{epochs}: mean loss {total_loss / n_batches:.4f}")
    policy.eval()


def run_grpo(
    n_train: int = 200,
    n_eval: int = 50,
    steps: int = 12,
    batch_prompts: int = 4,
    group_size: int = GROUP_SIZE,
    max_new_tokens: int = 24,
    lr: float = 1e-5,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    torch.set_num_threads(4)
    tok = load_tokenizer(POLICY_MODEL)
    tok.padding_side = "right"

    train_problems = make_math_problems(n_train, seed=seed)
    eval_problems = make_math_problems(n_eval, seed=seed + 1000)

    policy = load_causal_lm(POLICY_MODEL)
    trainable = freeze_all_but_last_blocks(policy, n_blocks=2)
    print(f"GRPO policy: {len(trainable)} trainable tensors (last 2 blocks)")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)

    base = eval_math(policy, tok, eval_problems, max_new_tokens)
    print(
        f"  base model: mean reward {base['mean_reward']:.3f}, "
        f"exact match {base['exact_match_rate']:.3f}"
    )

    sft_warm_start(policy, tok, train_problems, epochs=3, lr=2e-5, batch_size=16, seed=seed)

    before = eval_math(policy, tok, eval_problems, max_new_tokens)
    print(
        f"  after SFT / before GRPO: mean reward {before['mean_reward']:.3f}, "
        f"exact match {before['exact_match_rate']:.3f}"
    )

    for step in range(steps):
        batch = train_problems[step * batch_prompts:(step + 1) * batch_prompts]
        if not batch:
            break
        # 1. Sample a group of completions per prompt.
        group_prompts = [p["prompt"] for p in batch for _ in range(group_size)]
        policy.eval()
        full_texts, completions = generate_completions(
            policy, tok, group_prompts, max_new_tokens=max_new_tokens, temperature=1.0
        )
        prompt_lens = [len(tok(p)["input_ids"]) for p in group_prompts]
        # 2. Verifiable rewards.
        rewards = torch.tensor(
            [
                math_reward(c, batch[i // group_size]["answer"])
                for i, c in enumerate(completions)
            ],
            dtype=torch.float32,
        )
        # 3. Group-relative advantages.
        adv = rewards.clone()
        for g in range(len(batch)):
            grp = rewards[g * group_size:(g + 1) * group_size]
            adv[g * group_size:(g + 1) * group_size] = (grp - grp.mean()) / (grp.std() + 1e-4)
        # Groups where every sample scored the same give zero advantage and
        # therefore no gradient; report how often that happened.
        zero_groups = sum(
            1
            for g in range(len(batch))
            if rewards[g * group_size:(g + 1) * group_size].std() == 0
        )
        # 4. Old + ref log-probs (reference loaded briefly, then freed).
        with torch.no_grad():
            old_logp, mask = batch_logprobs(policy, tok, full_texts, prompt_lens)
        ref_model = load_causal_lm(POLICY_MODEL)
        ref_model.eval()
        with torch.no_grad():
            ref_logp, _ = batch_logprobs(ref_model, tok, full_texts, prompt_lens)
        del ref_model
        kl_per_seq = ((old_logp - ref_logp) * mask).sum(dim=1)
        # 5. Clipped GRPO update (KL penalty vs the reference included).
        loss = clipped_policy_update(
            policy,
            tok,
            full_texts,
            prompt_lens,
            old_logp,
            mask,
            adv,
            opt,
            CLIP_EPS,
            kl_beta=KL_BETA,
            kl_per_seq=kl_per_seq,
        )
        print(
            f"  step {step + 1}/{steps}: mean group reward {rewards.mean():.3f}, "
            f"zero-advantage groups {zero_groups}/{len(batch)}, loss {loss:.4f}"
        )
        del old_logp, ref_logp

    after = eval_math(policy, tok, eval_problems, max_new_tokens)
    print(
        f"  after GRPO: mean reward {after['mean_reward']:.3f}, "
        f"exact match {after['exact_match_rate']:.3f}"
    )

    out_dir = CHECKPOINTS_DIR / "grpo_policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out_dir / "pytorch_model.bin")

    results = {
        "part": "grpo",
        "task": "single-digit addition/subtraction, answer after 'Answer:'",
        "group_size": group_size,
        "steps": steps,
        "n_train": n_train,
        "n_eval": n_eval,
        "kl_beta": KL_BETA,
        "clip_eps": CLIP_EPS,
        "sft_epochs": 3,
        "sft_lr": 2e-5,
        "mean_reward_base": round(base["mean_reward"], 4),
        "exact_match_base": round(base["exact_match_rate"], 4),
        "any_number_base": round(base["any_number_rate"], 4),
        "mean_reward_before": round(before["mean_reward"], 4),
        "mean_reward_after": round(after["mean_reward"], 4),
        "exact_match_before": round(before["exact_match_rate"], 4),
        "exact_match_after": round(after["exact_match_rate"], 4),
        "any_number_before": round(before["any_number_rate"], 4),
        "any_number_after": round(after["any_number_rate"], 4),
        "sample_prompt": eval_problems[0]["prompt"],
        "sample_completion_base": base["completions"][0],
        "sample_completion_before": before["completions"][0],
        "sample_completion_after": after["completions"][0],
        "sample_answer": eval_problems[0]["answer"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "grpo.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    print(json.dumps(run_grpo(), indent=2))
