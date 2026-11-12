"""Data loading for all three demos.

Real public data first, with a clearly-labelled synthetic fallback when a
download fails:
- RLHF preferences: Anthropic HH-RLHF (streamed, first N pairs), cached to
  ``data/hh_rlhf_subset.csv`` so later runs are deterministic and offline.
- GRPO task: short addition/subtraction problems generated with a seeded
  RNG. The task itself is synthetic by design (a verifiable-reward task);
  the labels are exact arithmetic, not model output.
- Distillation task: SST-2 sentiment (GLUE), cached to ``data/sst2_subset.csv``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .common import DATA_DIR

HH_CACHE = DATA_DIR / "hh_rlhf_subset.csv"
SST2_CACHE = DATA_DIR / "sst2_subset.csv"

# ---------------------------------------------------------------------------
# HH-RLHF preferences
# ---------------------------------------------------------------------------

_SYNTH_PROMPTS = [
    "How do I bake a chocolate cake?",
    "What is the capital of France?",
    "Explain photosynthesis to a 10 year old.",
    "How can I improve my sleep?",
    "What is the difference between a virus and a bacterium?",
    "Give me tips for learning Spanish.",
    "How do I change a flat tire?",
    "What causes the seasons?",
    "How can I save money on groceries?",
    "Explain what a black hole is.",
    "What are the benefits of regular exercise?",
    "How does a refrigerator work?",
    "What is the best way to learn to code?",
    "How do I write a good resume?",
    "What is climate change?",
    "How can I be more productive at work?",
    "What is the speed of light?",
    "How do I start a vegetable garden?",
    "Explain the water cycle.",
    "What is an API?",
]

_CHOSEN_TEMPLATES = [
    "Here is a helpful answer: {topic}. The key points are to start with the basics, follow the steps in order, and check your work as you go. Let me know if you want more detail on any part.",
    "Good question. For {topic}, the short version is that it comes down to a few core ideas. First, understand the fundamentals. Second, practise with small examples. Third, build up gradually. Happy to go deeper on any step.",
]

_REJECTED_TEMPLATES = [
    "I don't know. Figure it out yourself.",
    "Who cares? Just google it.",
    "No. That is wrong and you should not ask that.",
    "Whatever. It does not matter.",
]


def synthetic_preferences(n_pairs: int, seed: int = 42) -> list[dict]:
    """Clearly-labelled synthetic preference pairs (fallback only).

    Chosen responses are helpful and on-topic; rejected ones are dismissive
    or unhelpful. Used only when the HH-RLHF download fails.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    for i in range(n_pairs):
        prompt = _SYNTH_PROMPTS[i % len(_SYNTH_PROMPTS)]
        chosen = rng.choice(_CHOSEN_TEMPLATES).format(topic=prompt.lower().rstrip("?"))
        rejected = rng.choice(_REJECTED_TEMPLATES)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "synthetic"})
    return pairs


def _split_hh_text(text: str) -> tuple[str, str]:
    """Split an HH-RLHF turn into (prompt, response)."""
    marker = "\n\nAssistant:"
    if marker in text:
        prompt, response = text.split(marker, 1)
        return prompt.strip(), response.strip()
    return text.strip(), ""


def load_hh_subset(n_train: int = 240, n_eval: int = 60, seed: int = 42) -> list[dict]:
    """Load HH-RLHF preference pairs, or the cached / synthetic fallback.

    Returns a list of {"prompt", "chosen", "rejected", "source"} dicts.
    """
    total = n_train + n_eval
    if HH_CACHE.exists():
        with open(HH_CACHE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) >= total:
            return rows[:total]
    try:
        from datasets import load_dataset

        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        pairs: list[dict] = []
        for ex in ds:
            prompt_c, chosen = _split_hh_text(ex["chosen"])
            prompt_r, rejected = _split_hh_text(ex["rejected"])
            prompt = prompt_c or prompt_r
            if not prompt or not chosen or not rejected:
                continue
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "hh-rlhf"})
            if len(pairs) >= total:
                break
        if len(pairs) < total:
            raise RuntimeError(f"only {len(pairs)} HH-RLHF pairs streamed")
    except Exception as exc:  # noqa: BLE001 - fallback is the point
        print(f"HH-RLHF download failed ({exc}); using labelled synthetic preferences.")
        pairs = synthetic_preferences(total, seed=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HH_CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "chosen", "rejected", "source"])
        w.writeheader()
        w.writerows(pairs)
    return pairs


# ---------------------------------------------------------------------------
# GRPO verifiable-reward task (generated arithmetic)
# ---------------------------------------------------------------------------

def make_math_problems(n: int, seed: int = 42, digits: int = 1) -> list[dict]:
    """Generate short addition/subtraction problems with exact answers.

    ``digits=1`` (default): single-digit operands (2-9). The base policy
    essentially never samples a correct two-digit answer by chance, so
    GRPO gets no advantage signal from two-digit problems (first attempt:
    exact match stayed 0.000 after 8 steps). Single-digit answers are
    sampled occasionally, which is what makes the group-relative signal
    usable. ``digits=2`` gives two-digit operands (11-99).
    """
    rng = random.Random(seed)
    lo, hi = (2, 9) if digits == 1 else (11, 99)
    problems: list[dict] = []
    for _ in range(n):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if rng.random() < 0.5:
            problems.append({"prompt": f"Question: What is {a} + {b}?\nAnswer:", "answer": a + b})
        else:
            hi_, lo_ = max(a, b), min(a, b)
            problems.append({"prompt": f"Question: What is {hi_} - {lo_}?\nAnswer:", "answer": hi_ - lo_})
    return problems


def math_reward(completion: str, answer: int) -> float:
    """Verifiable reward for a math completion.

    1.0  the completion starts with the correct number (right format + answer)
    0.2  some integer appears, but the first one is wrong
    0.0  no integer at all

    Deliberately no "correct number anywhere" tier: an earlier version paid
    0.6 for the answer appearing anywhere, and GRPO learned to enumerate
    numbers ("1. 2. 3. ... 11.") so the answer was present by construction
    while exact match barely moved (0.100 -> 0.120). The first-number-only
    rule closes that exploit.
    """
    import re

    nums = re.findall(r"-?\d+", completion)
    if not nums:
        return 0.0
    try:
        first = int(nums[0])
    except ValueError:
        return 0.0
    return 1.0 if first == answer else 0.2


def math_exact_match(completion: str, answer: int) -> bool:
    import re

    nums = re.findall(r"-?\d+", completion)
    return bool(nums) and int(nums[0]) == answer


# ---------------------------------------------------------------------------
# SST-2 sentiment for distillation
# ---------------------------------------------------------------------------

_SYNTH_SENTIMENT = [
    ("I loved this movie, it was fantastic and moving.", 1),
    ("A wonderful film with great acting and a smart script.", 1),
    ("This was a delight from start to finish.", 1),
    ("Brilliant, funny, and surprisingly touching.", 1),
    ("One of the best films I have seen in years.", 1),
    ("I hated this movie, it was boring and slow.", 0),
    ("A terrible film with wooden acting and a weak plot.", 0),
    ("Painfully dull and a waste of time.", 0),
    ("The worst movie I have seen, unfunny and messy.", 0),
    ("Disappointing, confusing, and far too long.", 0),
]


def synthetic_sentiment(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        text, label = _SYNTH_SENTIMENT[i % len(_SYNTH_SENTIMENT)]
        rows.append({"text": text, "label": label, "source": "synthetic"})
    rng.shuffle(rows)
    return rows


def _fetch_hf_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    """Fetch rows from the HF datasets-server rows API (plain JSON)."""
    import json
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "rlhf-demo/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    return [r["row"] for r in payload["rows"]]


def load_sst2_subset(n_train: int = 500, n_eval: int = 200, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Load SST-2 train/validation subsets, cached to CSV, synthetic fallback.

    Fetches real SST-2 rows through the HF datasets-server rows API (no
    full-dataset download), then shuffles with a fixed seed.
    """
    if SST2_CACHE.exists():
        with open(SST2_CACHE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        train = [r for r in rows if r["split"] == "train"]
        ev = [r for r in rows if r["split"] == "eval"]
        if len(train) >= n_train and len(ev) >= n_eval:
            return (
                [{"text": r["text"], "label": int(r["label"]), "source": r["source"]} for r in train[:n_train]],
                [{"text": r["text"], "label": int(r["label"]), "source": r["source"]} for r in ev[:n_eval]],
            )
    try:
        rng = random.Random(seed)
        train_rows: list[dict] = []
        offset = 0
        while len(train_rows) < n_train:
            chunk = _fetch_hf_rows("nyu-mll/glue", "sst2", "train", offset, min(100, n_train - len(train_rows)))
            if not chunk:
                break
            train_rows.extend(chunk)
            offset += len(chunk)
        eval_rows: list[dict] = []
        offset = 0
        while len(eval_rows) < n_eval:
            chunk = _fetch_hf_rows("nyu-mll/glue", "sst2", "validation", offset, min(100, n_eval - len(eval_rows)))
            if not chunk:
                break
            eval_rows.extend(chunk)
            offset += len(chunk)
        if len(train_rows) < n_train or len(eval_rows) < n_eval:
            raise RuntimeError(f"fetched {len(train_rows)} train / {len(eval_rows)} eval rows")
        train = [
            {"text": r["sentence"].strip(), "label": int(r["label"]), "source": "sst2"}
            for r in train_rows
        ]
        ev = [
            {"text": r["sentence"].strip(), "label": int(r["label"]), "source": "sst2"}
            for r in eval_rows
        ]
        rng.shuffle(train)
        rng.shuffle(ev)
    except Exception as exc:  # noqa: BLE001 - fallback is the point
        print(f"SST-2 download failed ({exc}); using labelled synthetic sentiment data.")
        train = synthetic_sentiment(n_train, seed=seed)
        ev = synthetic_sentiment(n_eval, seed=seed + 1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SST2_CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "text", "label", "source"])
        w.writeheader()
        for r in train:
            w.writerow({"split": "train", **r})
        for r in ev:
            w.writerow({"split": "eval", **r})
    return train, ev
