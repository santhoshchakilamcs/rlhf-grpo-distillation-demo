"""Tests for data generation and the verifiable math reward."""

from rlhf_demo.data import (
    make_math_problems,
    math_exact_match,
    math_reward,
    synthetic_preferences,
    synthetic_sentiment,
)


def test_math_problems_deterministic():
    a = make_math_problems(50, seed=7)
    b = make_math_problems(50, seed=7)
    assert a == b
    c = make_math_problems(50, seed=8)
    assert a != c


def test_math_problems_answers_correct():
    for p in make_math_problems(200, seed=1):
        if "+" in p["prompt"]:
            nums = [int(x) for x in p["prompt"].split("What is ")[1].split("?")[0].split(" + ")]
            assert p["answer"] == nums[0] + nums[1]
        else:
            nums = [int(x) for x in p["prompt"].split("What is ")[1].split("?")[0].split(" - ")]
            assert p["answer"] == nums[0] - nums[1]


def test_math_reward_tiers():
    assert math_reward("64", 64) == 1.0
    assert math_reward("The answer is 64.", 64) == 1.0  # first number correct
    # No "correct number anywhere" tier: listing numbers to include the
    # answer by construction must not be rewarded (see data.math_reward).
    assert math_reward("Hmm, 12 or maybe 64?", 64) == 0.2
    assert math_reward("I think 12.", 64) == 0.2  # a number, wrong one
    assert math_reward("I do not know.", 64) == 0.0  # no number


def test_math_exact_match():
    assert math_exact_match("64", 64) is True
    assert math_exact_match("12 or 64", 64) is False  # first number must be right
    assert math_exact_match("no digits", 64) is False


def test_synthetic_preferences_shape():
    pairs = synthetic_preferences(20, seed=3)
    assert len(pairs) == 20
    for p in pairs:
        assert p["prompt"] and p["chosen"] and p["rejected"]
        assert p["chosen"] != p["rejected"]
        assert p["source"] == "synthetic"


def test_synthetic_sentiment_balanced():
    rows = synthetic_sentiment(10, seed=3)
    assert len(rows) == 10
    assert sum(r["label"] for r in rows) == 5
    assert all(r["source"] == "synthetic" for r in rows)
