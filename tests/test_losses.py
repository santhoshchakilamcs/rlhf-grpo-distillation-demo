"""Tests for the loss functions (Bradley-Terry, GRPO advantage, distill KL)."""

import torch
import torch.nn.functional as F


def bradley_terry_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def test_bradley_terry_loss_values():
    # Equal rewards -> -log(0.5) = 0.6931
    loss = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert abs(loss.item() - 0.6931) < 1e-3
    # Chosen clearly above rejected -> small loss
    loss = bradley_terry_loss(torch.tensor([5.0]), torch.tensor([-5.0]))
    assert loss.item() < 0.001
    # Chosen below rejected -> large loss
    loss = bradley_terry_loss(torch.tensor([-5.0]), torch.tensor([5.0]))
    assert loss.item() > 9.0


def group_advantages(rewards, group_size):
    adv = rewards.clone()
    for g in range(len(rewards) // group_size):
        grp = rewards[g * group_size:(g + 1) * group_size]
        adv[g * group_size:(g + 1) * group_size] = (grp - grp.mean()) / (grp.std() + 1e-4)
    return adv


def test_grpo_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    adv = group_advantages(rewards, 4)
    # First group: mean 0.25, so the 1.0 sample gets a positive advantage and
    # the 0.0 samples a negative one; advantages sum to ~0 within a group.
    assert adv[0] > 0
    assert all(a < 0 for a in adv[1:4])
    assert abs(adv[0:4].sum().item()) < 1e-3
    # Second group: all equal -> all advantages ~0.
    assert torch.allclose(adv[4:8], torch.zeros(4), atol=1e-3)


def test_distillation_kl_zero_when_teacher_equals_student():
    logits = torch.randn(4, 2)
    kl = F.kl_div(
        F.log_softmax(logits / 2.0, dim=-1),
        F.softmax(logits / 2.0, dim=-1),
        reduction="batchmean",
    )
    assert abs(kl.item()) < 1e-5


def test_distillation_kl_positive_when_different():
    t = torch.tensor([[5.0, -5.0]])
    s = torch.tensor([[-5.0, 5.0]])
    kl = F.kl_div(
        F.log_softmax(s / 2.0, dim=-1),
        F.softmax(t / 2.0, dim=-1),
        reduction="batchmean",
    )
    assert kl.item() > 1.0


def test_clipped_surrogate_bounds_ratio():
    ratio = torch.tensor([10.0, 0.01, 1.0])
    adv = torch.tensor([1.0, 1.0, -1.0])
    clipped = torch.clamp(ratio, 0.8, 1.2)
    per_token = torch.minimum(ratio * adv, clipped * adv)
    # Ratio 10 with positive advantage is clipped to 1.2; ratio 0.01 with
    # positive advantage stays 0.01 (minimum picks the smaller).
    assert abs(per_token[0].item() - 1.2) < 1e-6
    assert abs(per_token[1].item() - 0.01) < 1e-6


def test_repetition_penalty():
    from rlhf_demo.ppo import repetition_penalty

    assert repetition_penalty("Assistant: " * 12) > 0.9
    assert repetition_penalty("The cat sat on the mat and watched the birds.") < 0.3
    assert repetition_penalty("short") == 0.0
