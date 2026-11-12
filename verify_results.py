"""Re-run the evaluations from the saved checkpoints and compare against
results/*.json. This is the final verification pass: it does NOT retrain
anything, it only loads each checkpoint and re-scores the same held-out
sets the training runs used, so the numbers must match the saved results
(up to CPU floating-point noise).

Usage: PYTHONPATH=src .venv/bin/python verify_results.py
"""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForSequenceClassification

from rlhf_demo.common import (
    CHECKPOINTS_DIR,
    POLICY_MODEL,
    RESULTS_DIR,
    load_causal_lm,
    load_tokenizer,
    set_seed,
)
from rlhf_demo.data import load_hh_subset, load_sst2_subset, make_math_problems
from rlhf_demo.distill import accuracy, student_config
from rlhf_demo.grpo import eval_math
from rlhf_demo.ppo import ASSISTANT_MARKER, eval_policy_reward
from rlhf_demo.reward_model import load_reward_model, reward_scores

set_seed(42)
torch.set_num_threads(4)


def _check(name: str, key: str, got: float, want: float, tol: float = 0.01) -> bool:
    ok = abs(got - want) <= tol
    print(f"  {name} {key}: got {got:.4f}, saved {want:.4f}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> bool:
    all_ok = True

    # --- Reward model: held-out preference accuracy (pairs 320-400) ---
    print("reward model (checkpoint -> held-out 80 HH-RLHF pairs)")
    saved = json.load(open(RESULTS_DIR / "reward_model.json"))
    rm, tok_rm = load_reward_model()
    pairs = load_hh_subset(320, 80, seed=42)
    eval_pairs = pairs[320:]
    r_c = reward_scores(rm, tok_rm, [p["prompt"] + "\n\nAssistant: " + p["chosen"] for p in eval_pairs], max_len=160)
    r_r = reward_scores(rm, tok_rm, [p["prompt"] + "\n\nAssistant: " + p["rejected"] for p in eval_pairs], max_len=160)
    acc = sum(1 for a, b in zip(r_c, r_r) if a > b) / len(eval_pairs)
    all_ok &= _check("reward model", "held_out_accuracy", acc, saved["held_out_accuracy"])

    # --- PPO: greedy eval on the 24 held-out prompts ---
    print("PPO (checkpoint -> 24 held-out prompts, greedy)")
    saved = json.load(open(RESULTS_DIR / "ppo.json"))
    pairs = load_hh_subset(400, 60, seed=42)
    eval_prompts = [p["prompt"] + ASSISTANT_MARKER for p in pairs[432:456]]
    tok = load_tokenizer(POLICY_MODEL)
    tok.padding_side = "right"
    policy = load_causal_lm(POLICY_MODEL)
    policy.load_state_dict(torch.load(CHECKPOINTS_DIR / "ppo_policy" / "pytorch_model.bin", map_location="cpu", weights_only=True))
    policy.eval()
    res = eval_policy_reward(policy, tok, rm, eval_prompts, 32)
    all_ok &= _check("PPO", "mean_reward_after", res["mean_reward"], saved["mean_reward_after"])
    all_ok &= _check("PPO", "mean_reward_penalised_after", res["mean_reward_penalised"], saved["mean_reward_penalised_after"])

    # --- GRPO: greedy eval on 50 held-out single-digit problems ---
    print("GRPO (checkpoint -> 50 held-out problems, greedy)")
    saved = json.load(open(RESULTS_DIR / "grpo.json"))
    eval_problems = make_math_problems(50, seed=1042)
    gpolicy = load_causal_lm(POLICY_MODEL)
    gpolicy.load_state_dict(torch.load(CHECKPOINTS_DIR / "grpo_policy" / "pytorch_model.bin", map_location="cpu", weights_only=True))
    gpolicy.eval()
    gres = eval_math(gpolicy, tok, eval_problems, 24)
    all_ok &= _check("GRPO", "mean_reward_after", gres["mean_reward"], saved["mean_reward_after"])
    all_ok &= _check("GRPO", "exact_match_after", gres["exact_match_rate"], saved["exact_match_after"])

    # --- Distillation: SST-2 accuracy of the three saved classifiers ---
    print("distillation (checkpoints -> 200 held-out SST-2 rows)")
    saved = json.load(open(RESULTS_DIR / "distillation.json"))
    train_rows, eval_rows = load_sst2_subset(500, 200, seed=42)
    tok_t = load_tokenizer("gpt2")
    if tok_t.pad_token is None:
        tok_t.pad_token = tok_t.eos_token
    teacher = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2, dtype=torch.float32)
    teacher.config.pad_token_id = tok_t.pad_token_id
    teacher.load_state_dict(torch.load(CHECKPOINTS_DIR / "distillation" / "teacher.bin", map_location="cpu", weights_only=True))
    all_ok &= _check("distillation", "teacher_accuracy", accuracy(teacher, eval_rows, tok_t), saved["teacher_accuracy"])
    del teacher
    student = AutoModelForSequenceClassification.from_config(student_config())
    student.config.pad_token_id = tok_t.pad_token_id
    student.load_state_dict(torch.load(CHECKPOINTS_DIR / "distillation" / "student_distilled.bin", map_location="cpu", weights_only=True))
    all_ok &= _check("distillation", "student_distilled_accuracy", accuracy(student, eval_rows, tok_t), saved["student_distilled_accuracy"])
    del student
    student_ce = AutoModelForSequenceClassification.from_config(student_config())
    student_ce.config.pad_token_id = tok_t.pad_token_id
    student_ce.load_state_dict(torch.load(CHECKPOINTS_DIR / "distillation" / "student_ce_only.bin", map_location="cpu", weights_only=True))
    all_ok &= _check("distillation", "student_ce_only_accuracy", accuracy(student_ce, eval_rows, tok_t), saved["student_ce_only_accuracy"])

    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS MISMATCHED")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
