"""Run all three demos end to end and merge the results.

Each stage runs as its own subprocess (``python -m rlhf_demo.<stage>``)
so model weights are freed from memory between stages on a CPU-only box.
Every stage writes results/<stage>.json; this script merges them into
results/summary.json and prints it.

Usage:
    .venv/bin/python run_all.py [--skip reward_model|ppo|grpo|distillation]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS_DIR = ROOT / "results"
ORDER = ["reward_model", "ppo", "grpo", "distill"]
RESULT_FILE = {
    "reward_model": "reward_model.json",
    "ppo": "ppo.json",
    "grpo": "grpo.json",
    "distill": "distillation.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], choices=ORDER)
    args = ap.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    summary: dict = {}
    for stage in ORDER:
        if stage in args.skip:
            print(f"=== skipping {stage} ===")
            continue
        print(f"=== running {stage} ===", flush=True)
        cmd = [sys.executable, "-m", f"rlhf_demo.{stage}"]
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            print(f"!!! stage {stage} failed (exit {proc.returncode}); stopping.")
            return proc.returncode
        with open(RESULTS_DIR / RESULT_FILE[stage], encoding="utf-8") as f:
            summary[stage] = json.load(f)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
