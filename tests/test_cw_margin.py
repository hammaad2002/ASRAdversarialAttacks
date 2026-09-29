"""The CW margin refinement (Sec. III-C) really runs.

The margin stage only starts once the CTC stage has reached the target, and on the toy
model that only happens for a target the model already produces. Aiming at the clean
transcript makes stage one succeed immediately, so these tests exercise the refinement
loop itself. The ``margin stage skipped`` warning is turned into an error so a test can
never pass by silently skipping the loop it claims to cover.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from asr_attacks.attacks.cw import _cw_margin_stage, _margin_frame_loss, cw

# ``asr_attacks.attacks.cw`` the attribute is the function; the module is in ``sys.modules``.
cw_module = sys.modules["asr_attacks.attacks.cw"]

EPSILON = 0.05

pytestmark = pytest.mark.filterwarnings("error:CW margin stage skipped")


def _run(backend, audio, target, **overrides):
    kwargs = dict(
        loss="margin",
        targeted=True,
        target=target,
        epsilon=EPSILON,
        num_iter=6,
        check_every=2,
        early_stop=False,
        search_eps=False,
        nested=False,
        verbose=False,
    )
    kwargs.update(overrides)
    return cw(backend, audio, **kwargs)


@pytest.fixture
def margin_calls(monkeypatch):
    """Record the ``frame_weights`` of every margin-loss evaluation."""
    weights: list[torch.Tensor] = []
    real = cw_module._margin_frame_loss

    def spy(logits, alignment, frame_weights, kappa):
        weights.append(frame_weights.detach().clone())
        return real(logits, alignment, frame_weights, kappa)

    monkeypatch.setattr(cw_module, "_margin_frame_loss", spy)
    return weights


@pytest.mark.parametrize("search_eps", [False, True], ids=["fixed-bound", "bound-search"])
@pytest.mark.parametrize("kappa", [0.0, 2.0])
def test_margin_stage_runs_and_respects_the_bound(backend, audio, margin_calls, search_eps, kappa):
    out = _run(backend, audio, backend.decode(audio), search_eps=search_eps, kappa=kappa)

    assert len(margin_calls) == 6
    assert isinstance(out, np.ndarray)
    assert out.shape == audio.numpy().shape
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0 + 1e-6
    assert np.abs(out - audio.numpy()).max() <= EPSILON + 1e-6


def test_margin_stage_returns_a_tensor_on_request(backend, audio, margin_calls):
    out = _run(backend, audio, backend.decode(audio), return_tensor=True)

    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()
    assert margin_calls


def test_margin_stage_honours_a_db_bound(backend, audio, margin_calls):
    from asr_attacks.metrics import db_distortion

    out = _run(backend, audio, backend.decode(audio), epsilon=1.0, db_bound=-25.0)

    assert margin_calls
    assert np.isfinite(out).all()
    assert db_distortion(audio, out) <= -25.0 + 1e-3


def test_margin_stage_on_a_silence_target(silent_backend, audio, margin_calls):
    assert silent_backend.decode(audio).strip("| ") == ""

    out = _run(silent_backend, audio, "")

    assert len(margin_calls) == 6
    assert np.isfinite(out).all()
    assert np.abs(out - audio.numpy()).max() <= EPSILON + 1e-6


def test_margin_stage_on_a_silence_target_returns_a_tensor(silent_backend, audio, margin_calls):
    out = _run(silent_backend, audio, "", return_tensor=True)

    assert margin_calls
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-6


def test_margin_early_stop_ends_once_the_alignment_holds(backend, audio, margin_calls):
    out = _run(backend, audio, backend.decode(audio), num_iter=50, early_stop=True)

    assert 1 <= len(margin_calls) < 50
    assert np.isfinite(out).all()


def test_margin_bound_search_shrinks_the_box_after_a_matching_alignment(
    backend, audio, monkeypatch
):
    bounds: list[float] = []
    real = cw_module._project_inplace

    def spy(adversarial, original, epsilon):
        bounds.append(float(epsilon))
        return real(adversarial, original, epsilon)

    monkeypatch.setattr(cw_module, "_project_inplace", spy)
    _run(backend, audio, backend.decode(audio), search_eps=True, num_iter=6, check_every=2)

    # Only the margin stage projects here: one projection per step, plus one shrink after each
    # check at which the greedy alignment still matches.
    assert len(bounds) > 6
    assert min(bounds) < bounds[0]


def test_margin_doubles_the_weight_of_frames_that_lost_their_alignment(
    backend, audio, margin_calls
):
    # The stage starts far from the original, so the L2 pull under a huge step size flips
    # greedy labels: some frames are "wrong" at the first check.
    # Called directly: the same step size would also stop stage one from reaching the target.
    generator = torch.Generator().manual_seed(3)
    stage_one = audio + 0.3 * torch.randn(audio.shape, generator=generator)
    _cw_margin_stage(
        backend,
        audio,
        stage_one,
        epsilon=1.0,
        c=1.0,
        learning_rate=0.5,
        num_iter=4,
        decrease_factor_eps=0.8,
        check_every=1,
        optimizer="adam",
        search_eps=True,
        early_stop=False,
        nested=False,
        verbose=False,
        kappa=0.0,
    )

    assert margin_calls[0].max().item() == pytest.approx(1.0)
    assert max(w.max().item() for w in margin_calls[1:]) >= 2.0


def test_margin_frame_loss_matches_a_hand_computation():
    logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 1.0, 2.5], [1.0, 2.0, 1.5]]])
    alignment = torch.tensor([0, 2, 1])
    weights = torch.tensor([1.0, 2.0, 4.0])

    # Frame 0: best wrong = 1.0, correct = 3.0 -> relu(-2 + kappa)
    # Frame 1: best wrong = 1.0, correct = 2.5 -> relu(-1.5 + kappa)
    # Frame 2: best wrong = 1.5, correct = 2.0 -> relu(-0.5 + kappa)
    assert _margin_frame_loss(logits, alignment, weights, 0.0).item() == pytest.approx(0.0)
    expected = 1.0 * 0.0 + 2.0 * 0.5 + 4.0 * 1.5
    assert _margin_frame_loss(logits, alignment, weights, 2.0).item() == pytest.approx(expected)


def test_margin_frame_loss_pushes_the_correct_label_up_and_the_runner_up_down():
    logits = torch.tensor([[[1.0, 2.0, 0.0]]], requires_grad=True)
    loss = _margin_frame_loss(logits, torch.tensor([0]), torch.tensor([1.0]), 0.0)
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert logits.grad.tolist() == [[[-1.0, 1.0, 0.0]]]


def test_margin_frame_loss_rejects_an_alignment_of_the_wrong_length():
    logits = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="Alignment length"):
        _margin_frame_loss(logits, torch.zeros(3, dtype=torch.long), torch.ones(3), 0.0)


def test_margin_is_skipped_with_a_warning_when_ctc_misses_the_target(backend, audio):
    with pytest.warns(UserWarning, match="CW margin stage skipped"):
        out = _run(backend, audio, "CCCCCCCCCC", num_iter=1)
    assert np.isfinite(out).all()
