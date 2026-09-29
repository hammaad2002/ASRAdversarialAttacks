"""Every attack mode returns finite, in-range audio of the input's shape.

Imperceptible runs the real psychoacoustic masking (no mocks) and the robust modes use
user-supplied impulse responses, so nothing here needs librosa or pyroomacoustics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_attacks.attacks.cw import cw
from asr_attacks.attacks.imperceptible import imperceptible
from asr_attacks.metrics import db_distortion

EPSILON = 0.05
ROBUST_DELTA = 300.0 / 32768.0


def _assert_valid(out, audio: torch.Tensor, *, changed: bool = True) -> None:
    reference = audio.numpy()
    assert isinstance(out, np.ndarray)
    assert out.shape == reference.shape
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0 + 1e-6
    if changed:
        assert np.abs(out - reference).max() > 0.0


IMPERCEPTIBLE_CASES = {
    "offline-targeted": ("imperceptible", {"targeted": True, "target": "AB"}),
    "offline-untargeted": ("imperceptible", {"targeted": False}),
    "offline-untargeted-label": ("imperceptible", {"targeted": False, "label": "CAB"}),
    "offline-silence": ("imperceptible", {"targeted": True, "target": ""}),
    "robust-targeted": ("robust", {"targeted": True, "target": "AB"}),
    "robust-untargeted": ("robust", {"targeted": False}),
    "robust-untargeted-label": ("robust", {"targeted": False, "label": "CAB"}),
    "robust-silence": ("robust", {"targeted": True, "target": ""}),
    "imperceptible-robust-targeted": ("imperceptible_robust", {"targeted": True, "target": "AB"}),
    "imperceptible-robust-untargeted": ("imperceptible_robust", {"targeted": False}),
    "imperceptible-robust-label": (
        "imperceptible_robust",
        {"targeted": False, "label": "CAB"},
    ),
    "imperceptible-robust-silence": ("imperceptible_robust", {"targeted": True, "target": ""}),
}


@pytest.mark.parametrize(("mode", "kwargs"), IMPERCEPTIBLE_CASES.values(), ids=IMPERCEPTIBLE_CASES)
def test_imperceptible_modes_return_finite_in_range_audio(backend, long_audio, rooms, mode, kwargs):
    out = imperceptible(
        backend,
        long_audio,
        mode=mode,
        rooms=rooms if mode != "imperceptible" else None,
        epsilon=EPSILON,
        num_iter1=3,
        num_iter2=3,
        num_iter_r1=3,
        num_iter_r2=3,
        num_iter_ir1=3,
        num_iter_ir2=3,
        check_every=1,
        early_stop_cw=False,
        search_eps_cw=False,
        nested=False,
        verbose=False,
        **kwargs,
    )

    _assert_valid(out, long_audio)
    if mode != "imperceptible":
        # Robust stages stay inside the bound found by R1 plus the R2 slack.
        assert np.abs(out - long_audio.numpy()).max() <= EPSILON + ROBUST_DELTA + 1e-6


def test_imperceptible_with_bound_search_and_early_stop_flags(backend, long_audio):
    """The bound-shrinking search path (default in the paper) also stays finite."""
    out = imperceptible(
        backend,
        long_audio,
        "AB",
        epsilon=EPSILON,
        num_iter1=6,
        num_iter2=3,
        check_every=2,
        search_eps_cw=True,
        early_stop_cw=False,
        nested=False,
        verbose=False,
    )
    _assert_valid(out, long_audio)


CW_CASES = {
    "ctc-targeted": {"loss": "ctc", "targeted": True, "target": "AB"},
    "ctc-untargeted": {"loss": "ctc", "targeted": False},
    "ctc-untargeted-label": {"loss": "ctc", "targeted": False, "label": "CAB"},
    "ctc-silence": {"loss": "ctc", "targeted": True, "target": ""},
}
# ``loss="margin"`` only starts after the CTC stage hit its target, so it lives in
# ``test_cw_margin.py`` where the target is reachable and a skipped stage is an error.


@pytest.mark.parametrize("search_eps", [False, True], ids=["fixed-bound", "bound-search"])
@pytest.mark.parametrize("kwargs", CW_CASES.values(), ids=CW_CASES)
def test_cw_losses_and_targets_return_finite_in_range_audio(backend, audio, kwargs, search_eps):
    out = cw(
        backend,
        audio,
        epsilon=EPSILON,
        num_iter=6,
        check_every=2,
        early_stop=False,
        search_eps=search_eps,
        nested=False,
        verbose=False,
        **kwargs,
    )

    _assert_valid(out, audio)
    assert np.abs(out - audio.numpy()).max() <= EPSILON + 1e-6


def test_cw_db_bound_limits_the_distortion(backend, audio):
    out = cw(
        backend,
        audio,
        db_bound=-25.0,
        loss="ctc",
        targeted=True,
        target="AB",
        num_iter=6,
        early_stop=False,
        search_eps=False,
        nested=False,
        verbose=False,
    )

    _assert_valid(out, audio)
    assert db_distortion(audio, out) <= -25.0 + 1e-3


def test_cw_return_tensor_stays_finite_for_silence(backend, audio):
    out = cw(
        backend,
        audio,
        loss="ctc",
        targeted=True,
        target="",
        epsilon=EPSILON,
        num_iter=4,
        early_stop=False,
        search_eps=False,
        nested=False,
        verbose=False,
        return_tensor=True,
    )
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-6
