"""Targets that cannot fit in the audio fail loudly instead of producing NaN audio."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_attacks.attacks.common import ctc_loss, min_ctc_frames
from asr_attacks.attacks.cw import cw, imperceptible
from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.iterative import bim, pgd


def _logits(frames: int, vocab: int = 5) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, frames, vocab, requires_grad=True)


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([], 0),
        ([1], 1),
        ([1, 2, 3], 3),
        ([1, 1], 3),  # a blank must separate the two identical tokens
        ([1, 1, 1], 5),
        ([1, 2, 1], 3),
    ],
)
def test_min_ctc_frames(ids, expected):
    assert min_ctc_frames(torch.tensor(ids, dtype=torch.long)) == expected


def test_min_ctc_frames_uses_worst_row_of_a_batch():
    ids = torch.tensor([[1, 2, 3], [1, 1, 1]])
    assert min_ctc_frames(ids) == 5


def test_ctc_loss_rejects_target_longer_than_frames():
    with pytest.raises(ValueError, match="at least 6 logit frames"):
        ctc_loss(_logits(4), torch.tensor([1, 2, 3, 1, 2, 3]), blank_id=0)


def test_ctc_loss_counts_repeats_that_need_a_blank():
    # Three identical tokens need 5 frames; 4 is one short even though 3 <= 4.
    with pytest.raises(ValueError, match="at least 5 logit frames"):
        ctc_loss(_logits(4), torch.tensor([1, 1, 1]), blank_id=0)


@pytest.mark.parametrize(("frames", "ids"), [(3, [1, 2, 3]), (5, [1, 1, 1]), (4, [1, 1, 2])])
def test_ctc_loss_accepts_exactly_tight_targets(frames, ids):
    logits = _logits(frames)
    loss = ctc_loss(logits, torch.tensor(ids), blank_id=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ctc_loss_accepts_empty_target():
    loss = ctc_loss(_logits(4), torch.zeros(0, dtype=torch.long), blank_id=0)
    assert torch.isfinite(loss)


def test_zero_infinity_keeps_numeric_infinities_out_of_the_gradient():
    """A class the model rules out entirely (-inf) must not poison the gradient."""
    logits = torch.zeros(1, 4, 5)
    logits[..., 1] = float("-inf")
    logits.requires_grad_(True)

    loss = ctc_loss(logits, torch.tensor([1, 2]), blank_id=0)

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.fixture
def short_audio() -> torch.Tensor:
    # TinyCTC emits one frame per sample, so this audio yields only 4 frames.
    torch.manual_seed(1)
    return torch.randn(1, 4) * 0.05


def test_attacks_raise_a_clear_error_for_an_oversized_target(backend, short_audio):
    too_long = "ABCAB"  # 5 tokens > 4 frames
    with pytest.raises(ValueError, match="logit frames"):
        fgsm(backend, short_audio, targeted=True, target=too_long)
    with pytest.raises(ValueError, match="logit frames"):
        bim(backend, short_audio, num_iter=1, targeted=True, target=too_long, verbose=False)
    with pytest.raises(ValueError, match="logit frames"):
        pgd(backend, short_audio, num_iter=1, targeted=True, target=too_long, verbose=False)
    with pytest.raises(ValueError, match="logit frames"):
        cw(
            backend,
            short_audio,
            num_iter=1,
            targeted=True,
            target=too_long,
            search_eps=False,
            verbose=False,
        )


def test_untargeted_label_longer_than_audio_is_rejected(backend, short_audio):
    with pytest.raises(ValueError, match="logit frames"):
        bim(backend, short_audio, num_iter=1, label="ABCAB", verbose=False)


def test_a_target_that_just_fits_yields_finite_audio(backend, short_audio):
    fits = "ABCA"  # 4 tokens, no repeats -> exactly 4 frames
    out = bim(backend, short_audio, epsilon=0.01, num_iter=3, targeted=True, target=fits)
    assert isinstance(out, np.ndarray)
    assert np.isfinite(out).all()
    out = cw(
        backend,
        short_audio,
        num_iter=3,
        targeted=True,
        target=fits,
        search_eps=False,
        early_stop=False,
        verbose=False,
    )
    assert np.isfinite(out).all()


def test_imperceptible_rejects_an_oversized_target_before_stage_two(backend):
    audio = torch.randn(1, 6) * 0.05
    with pytest.raises(ValueError, match="logit frames"):
        imperceptible(
            backend,
            audio,
            "ABCABCA",
            num_iter1=1,
            num_iter2=1,
            early_stop_cw=False,
            search_eps_cw=False,
            nested=False,
            verbose=False,
        )
