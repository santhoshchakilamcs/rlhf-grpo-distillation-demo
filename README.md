# RLHF, GRPO & Knowledge Distillation Demo

A personal project that implements the three core LLM post-training
techniques end to end on CPU with tiny models: a **reward model +
PPO (RLHF)**, **GRPO** on a verifiable-reward task, and **knowledge
distillation** of a larger teacher into a smaller student.

Built to cover the RLHF/GRPO and distillation gaps in my record for an
ML Engineer role in customer-assistance AI. It is a **demo, not
production RLHF**: small public models, a few hundred training examples,
a handful of update steps, all on CPU. Every number below was measured
by running the code on a 4-CPU / ~8 GB RAM box (Python 3.12, torch
2.15 CPU, transformers 5.22, datasets 5.2). No Airbnb data, no
production scale, no production RLHF is claimed anywhere.

## What each part does

### 1. Reward model + PPO (`src/rlhf_demo/reward_model.py`, `ppo.py`)

- **Reward model:** a scalar head on `distilgpt2` (82M), last 2
  transformer blocks trainable + head. Trained with a Bradley–Terry
  loss on **400 real Anthropic HH-RLHF preference pairs** (320 train /
  80 held out). Response-priority left truncation keeps the scored
  assistant response intact.
- **PPO:** a minimal from-scratch PPO loop (clipped surrogate,
  KL penalty vs a frozen reference, scalar baseline) — **not TRL**.
  Only the last 2 blocks of `distilgpt2` are trainable. Rollouts are
  scored by the reward model minus a **repetition penalty** (fraction
  of duplicate bigrams, weight 0.5), added after the first run showed
  the policy gaming the weak reward model (see "What did not work").
- Evaluated on 24 held-out HH-RLHF prompts, greedy decoding, reward
  scored by the reward model.

### 2. GRPO (`src/rlhf_demo/grpo.py`)

- Minimal from-scratch GRPO (DeepSeekMath-style): for each prompt, a
  group of **G = 4** sampled completions, programmatic verifiable
  reward, **group-relative advantage** `(r − group mean) / (group std
  + eps)`, clipped update, KL penalty vs a frozen reference.
- Task: single-digit addition/subtraction, prompt ends with
  `Answer:`, reward is **1.0** if the first number emitted is correct,
  **0.2** if any integer is emitted but the first is wrong, **0.0**
  otherwise.
- An **SFT warm-start** (3 epochs, loss on answer tokens only) comes
  first: the base model essentially never emits a bare number after
  `Answer:`, so without it GRPO finds almost no correct answers to
  amplify. Both the pre-SFT baseline and the post-SFT / pre-GRPO
  baseline are reported, so the GRPO delta is isolated.
- Evaluated on 50 held-out problems, greedy decoding.

### 3. Distillation (`src/rlhf_demo/distill.py`)

- Teacher: `gpt2` (124M) fully fine-tuned on **500 real SST-2**
  sentiment examples (3 epochs, head at a higher LR — a randomly
  initialised head barely moves at the body LR).
- Student: a from-config GPT-2 (4 layers, hidden 384, ~26M params,
  randomly initialised) trained two ways on the same 500 examples:
  **(a)** distillation — KL to the teacher's soft labels at
  temperature 2, `α_CE = 0.3`; **(b)** cross-entropy-only baseline.
- Reports teacher vs distilled-student vs CE-only accuracy on 200
  held-out SST-2 rows, parameter counts, fp32 sizes, and mean
  per-example CPU latency.

## Measured results (all eval-mode, verified from saved checkpoints)

Re-measured from the saved checkpoints with `verify_results.py`
(8/8 checks pass) and reproduced by `pytest` (20 tests pass).

### Reward model

| | |
|---|---|
| Held-out preference accuracy | **0.550 (44/80)** |
| Mean chosen − rejected reward gap | **+0.110** |
| Train-subsample accuracy (60 pairs) | **0.683** |

Modest but real and above chance — a usable demo-scale reward signal,
not a strong one. The first two attempts scored 0.33 and 0.45
(before the truncation fix and the separate head learning rate).

### PPO

| | Before | After |
|---|---|---|
| Mean reward (24 held-out prompts, greedy) | 0.5404 | **0.4800** |
| Mean reward, repetition-penalised | 0.3115 | **0.2498** |
| Mean repetition penalty | 0.4577 | 0.4604 |
| Mean completion length (words) | 31.5 | 30.9 |

**PPO did not improve greedy completions at this scale.** Sampled
rollout rewards rose during training (mean penalised reward
0.010 → 0.518 across 8 iterations), but the greedy eval reward fell.
The likely cause is the weak 55%-accurate reward model: its signal is
noisy, and 8 iterations on 32 prompts is very little training. The
greedy after-completion is still a repeated phrase. This is reported
as-is; the value of this part is the working pipeline
(reward model → rollouts → KL-penalised clipped update → eval), not
a reward gain.

### GRPO

| | Base model | After SFT | After GRPO |
|---|---|---|---|
| Mean reward (50 held-out problems) | 0.128 | 0.280 | **0.280** |
| Exact match | 0.000 | 0.100 | **0.100** |
| Any-number rate | 0.640 | 1.000 | 1.000 |

GRPO works mechanically (group-relative advantage, KL penalty,
verifiable reward) but adds **nothing on the greedy eval at this
scale**: the SFT warm-start carries the whole gain (0.000 → 0.100
exact match) and GRPO holds it at 0.100. The correct-answer signal
is sparse — most sampled groups contain no correct answer, so most
steps give little or no gradient — and 12 steps is very short.
GRPO alone on the base model does not learn at all (exact match
stayed 0.000 after 8 steps on two-digit and 12 steps on
single-digit problems), which is why the SFT warm-start is part of
the pipeline. A standalone re-run of the SFT step gave 0.232 /
0.040, so the SFT outcome itself varies by about ±0.05 reward
across runs; the GRPO delta of 0.00 is inside that noise.

### Distillation

| | Teacher (gpt2, 124M) | Distilled student (~26M) | CE-only student (~26M) |
|---|---|---|---|
| SST-2 accuracy (200 held-out) | **0.765** | **0.560** | **0.510** |
| Parameters | 124,441,344 | 26,447,232 | 26,447,232 |
| fp32 size | 497.8 MB | 105.8 MB | 105.8 MB |
| CPU latency / example | 200.87 ms | 24.11 ms | — |

**Compression 4.71×, latency speedup 8.33×.** Distillation beats
the cross-entropy-only baseline by 5 points (0.560 vs 0.510): the
teacher's soft labels give generalisation signal that hard labels
alone do not — the CE-only student fits the training data (final
train loss 0.368, lower than the teacher's 0.398) but does not
generalise. The from-scratch student is still far from the teacher
(0.560 vs 0.765); that is the expected gap at this scale with 500
training examples.

## What did not work (documented, not hidden)

- **Reward hacking, unmitigated PPO:** the first PPO run's policy
  converged on repeating `Assistant:` to game the weak reward model
  (its reported reward rose while the text degenerated). A repetition
  penalty (weight 0.5) was added to rollout rewards; sampled-rollout
  repetition fell in most iterations, but greedy decoding of the 82M
  base model still falls into loops.
- **GRPO reward exploit:** an earlier reward tier paid 0.6 for the
  correct number appearing *anywhere*, and GRPO learned to enumerate
  numbers (`1. 2. 3. … 11.`) so the answer was present by
  construction. The tier was removed; the current reward only pays
  for the **first** number being correct.
- **Eval-mode bug (found during verification, Nov 12, 2026):** the
  first PPO/GRPO "after" numbers (0.8265 and 0.296) were measured
  with the policy in train mode — dropout active — and were not
  reproducible (the same checkpoint scored 0.57 and 0.93 on two
  train-mode runs). `greedy_completions` now forces eval mode, a
  regression test guards it, and the numbers above are the eval-mode
  re-measurements from the saved checkpoints. The intermediate
  attempt files (`results/grpo_2digit.json`,
  `results/grpo_no_sft.json`, `results/grpo_anywhere_tier.json`,
  `results/ppo_unmitigated.json`) keep their original train-mode
  numbers as attempt logs; only the final files were corrected.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # all four stages, ~1–2 h on a 4-CPU box
```

Each stage runs as its own subprocess so model weights are freed
between stages. Individual stages:

```bash
PYTHONPATH=src python -m rlhf_demo.reward_model   # reward model
PYTHONPATH=src python -m rlhf_demo.ppo             # PPO
PYTHONPATH=src python -m rlhf_demo.grpo            # GRPO (SFT + GRPO)
PYTHONPATH=src python -m rlhf_demo.distill         # distillation
```

Results land in `results/<stage>.json` (gitignored — they regenerate
on every run). Verify them without retraining:

```bash
PYTHONPATH=src python verify_results.py   # re-scores saved checkpoints
python -m pytest tests/ -q                # 20 tests
```

Models download from Hugging Face at runtime (`distilgpt2`,
`gpt2`); the two small data caches (`data/hh_rlhf_subset.csv`,
`data/sst2_subset.csv`) are real public data (Anthropic HH-RLHF and
GLUE SST-2) and regenerate on first run if deleted. No model
weights, databases, or large files are committed: `.gitignore`
excludes `.venv/`, `checkpoints/`, and `results/`.

## Layout

```
src/rlhf_demo/
  common.py        shared loading, seeding, freezing, paths
  data.py          HH-RLHF / SST-2 loading, math problems + reward
  reward_model.py  Bradley-Terry reward model
  policy.py        generation, log-probs, clipped PPO/GRPO update
  ppo.py           PPO loop + repetition-penalty eval
  grpo.py          SFT warm-start + GRPO loop
  distill.py       teacher / distilled / CE-only student
run_all.py         runs all stages, merges results/summary.json
verify_results.py  re-scores checkpoints, checks results/*.json
tests/             20 pytest tests (data, losses, models, smoke)
data/              small cached CSVs (real public data)
```

## Honest limits

- **Not TRL.** PPO and GRPO are minimal from-scratch
  implementations of the core algorithms (clipped surrogate,
  KL penalty, group-relative advantage). They are not TRL's
  `PPOTrainer` / `GRPOTrainer` and are not production RLHF.
- **Demo scale.** 82M–124M models, 320–500 training examples,
  8 PPO iterations, 12 GRPO steps, 3 epochs. A 55%-accurate reward
  model is a weak signal; results at this scale have meaningful
  run-to-run variation (see the SFT re-run note above).
- **No pretraining, no AutoGen, no production RL.** This project
  covers RLHF/GRPO and distillation only. It uses LangGraph-style
  multi-agent orchestration nowhere; that lives in other projects.
- **CPU only.** Everything runs on CPU; no GPU was used or needed.
