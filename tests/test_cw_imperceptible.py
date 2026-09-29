"""Tests for paper-faithful CW (Sec. III-B/C/F) and Qin et al. imperceptible modes."""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
import torch

from asr_attacks.attacks.common import ctc_loss
from asr_attacks.attacks.cw import (
    _bounded_search,
    _silence_loss,
    cw,
    imperceptible,
)
from asr_attacks.metrics import db_distortion
from asr_attacks.rooms import RoomSimulator, _fft_convolve_same_length
from asr_attacks.tensors import prepare_audio

_cw_module = sys.modules["asr_attacks.attacks.cw"]


@pytest.fixture
def patch_psycho(monkeypatch):
    """Avoid librosa/numba in CI: stage-2 only needs a finite masking penalty."""

    def fake_threshold(waveform, sample_rate=16000, **_kwargs):
        n = max(int(np.asarray(waveform).size) // 512, 1)
        theta = np.ones((n, 1025), dtype=np.float64)
        return theta, 1.0

    def fake_psd(delta, original_max_psd, device, **_kwargs):
        time = max(delta.shape[-1] // 512, 1)
        return torch.zeros(1, 1025, time, dtype=torch.float64, device=device)

    monkeypatch.setattr(_cw_module, "compute_masking_threshold", fake_threshold)
    monkeypatch.setattr(_cw_module, "psd_transform", fake_psd)


def test_cw_objective_is_squared_l2_plus_c_ctc(backend, audio):
    original = prepare_audio(audio, backend.device).detach()
    delta = torch.full_like(original, 0.01, requires_grad=True)
    target = "AB"
    ids = backend.encode(target)
    c = 2.5

    # Isolated squared-L2 term: grad must be exactly 2*delta.
    torch.sum(delta**2).backward()
    assert delta.grad is not None
    torch.testing.assert_close(delta.grad, 2.0 * delta.detach())

    delta2 = torch.full_like(original, 0.01, requires_grad=True)
    loss = torch.sum(delta2**2) + c * ctc_loss(
        backend.logits(original + delta2), ids, backend.blank_id
    )
    loss.backward()
    assert delta2.grad is not None
    # With c>0 the CTC contribution moves the gradient away from pure 2*delta.
    assert not torch.allclose(delta2.grad, 2.0 * delta2.detach())

    result = cw(
        backend,
        audio,
        epsilon=0.05,
        c=1.0,
        num_iter=2,
        early_stop=False,
        search_eps=False,
        targeted=True,
        target=target,
        optimizer="sgd",
        learning_rate=0.0,
        nested=False,
        verbose=False,
    )
    assert isinstance(result, np.ndarray)


def test_cw_search_eps_respects_bound_and_db_bound(backend, audio):
    peak = float(audio.abs().max())
    db = -20.0
    expected_eps = peak * (10.0 ** (db / 20.0))
    adv = cw(
        backend,
        audio,
        db_bound=db,
        c=1.0,
        num_iter=5,
        check_every=2,
        decrease_factor_eps=0.8,
        search_eps=True,
        early_stop=False,
        targeted=True,
        target="A",
        optimizer="adam",
        learning_rate=1e-2,
        nested=False,
        verbose=False,
    )
    assert np.abs(adv - audio.numpy()).max() <= expected_eps + 1e-5

    # When search_eps never shrinks below the start, max|delta| <= epsilon.
    epsilon = 0.04
    adv2 = cw(
        backend,
        audio,
        epsilon=epsilon,
        num_iter=4,
        search_eps=True,
        check_every=10,
        early_stop=False,
        targeted=False,
        nested=False,
        verbose=False,
    )
    assert np.abs(adv2 - audio.numpy()).max() <= epsilon + 1e-5


def test_cw_invalid_optimizer_raises(backend, audio):
    with pytest.raises(ValueError, match="optimizer"):
        cw(backend, audio, num_iter=1, optimizer="rmsprop", early_stop=False, search_eps=False)


def test_cw_margin_untargeted_raises_and_alignment_length(backend, audio):
    with pytest.raises(ValueError, match="targeted"):
        cw(
            backend,
            audio,
            loss="margin",
            targeted=False,
            num_iter=1,
            early_stop=False,
            search_eps=False,
        )

    stage = cw(
        backend,
        audio,
        loss="ctc",
        targeted=True,
        target="A",
        num_iter=3,
        early_stop=False,
        search_eps=False,
        nested=False,
        verbose=False,
        return_tensor=True,
    )
    assert isinstance(stage, torch.Tensor)
    logits = backend.logits(stage)
    alignment = torch.argmax(logits[0], dim=-1)
    assert alignment.numel() == logits.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cw(
            backend,
            audio,
            loss="margin",
            targeted=True,
            target="A",
            num_iter=2,
            early_stop=False,
            search_eps=False,
            nested=False,
            verbose=False,
        )


def test_cw_silence_target_and_db_distortion(backend, audio):
    ids = backend.silence_ids()
    assert backend.blank_id in ids
    assert backend._char_to_id["|"] in ids  # type: ignore[attr-defined]

    original = prepare_audio(audio, backend.device)
    loss = _silence_loss(backend.logits(original), ids)
    assert torch.isfinite(loss)

    adv = cw(
        backend,
        audio,
        targeted=True,
        target="",
        num_iter=3,
        early_stop=False,
        search_eps=False,
        nested=False,
        verbose=False,
    )
    assert np.isfinite(adv).all()

    x = np.array([0.0, 0.5, -0.25], dtype=np.float64)
    a = np.array([0.0, 0.6, -0.25], dtype=np.float64)
    # max|delta|=0.1, max|x|=0.5 → 20*log10(0.1)-20*log10(0.5)
    expected = 20.0 * np.log10(0.1) - 20.0 * np.log10(0.5)
    assert db_distortion(x, a) == pytest.approx(expected)


def test_imperceptible_stage1_bound_and_no_l2(backend, audio, patch_psycho):
    original = prepare_audio(audio, backend.device).detach()
    target = "A"
    eps = 0.05
    stage, final_eps = _bounded_search(
        backend,
        original,
        target,
        targeted=True,
        epsilon=eps,
        learning_rate=1e-2,
        num_iter=6,
        decrease_factor_eps=0.8,
        check_every=2,
        optimizer="sgd",
        search_eps=True,
        early_stop=False,
        nested=False,
        verbose=False,
        desc=None,
        use_l2=False,
        c=1.0,
        sign_grad=True,
    )
    assert float((stage - original).abs().max()) <= eps + 1e-5
    assert final_eps <= eps + 1e-12

    with pytest.warns(DeprecationWarning, match="deprecated"):
        out = imperceptible(
            backend,
            audio,
            target,
            epsilon=0.05,
            c=1e-4,
            num_iter1=3,
            num_iter2=4,
            learning_rate1=1e-2,
            learning_rate2=1e-3,
            early_stop_cw=False,
            search_eps_cw=False,
            nested=False,
            verbose=False,
            sample_rate=16000,
        )
    assert out.max() <= 1.0 + 1e-5
    assert out.min() >= -1.0 - 1e-5


def test_room_simulator_from_rirs_length_and_identity():
    signal = torch.tensor([1.0, 0.5, -0.25, 0.0, 0.125])
    rir = np.array([0.5, 0.25, 0.125], dtype=np.float64)
    rooms = RoomSimulator.from_rirs([rir])
    out = rooms.sample()(signal)
    assert out.shape == signal.shape
    expected = np.convolve(signal.numpy(), rir, mode="full")[: signal.numel()]
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    identity = RoomSimulator.from_rirs([np.array([1.0])])
    torch.testing.assert_close(identity.sample()(signal), signal, atol=1e-6, rtol=0)

    # Delta RIR via the low-level helper.
    delta = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(
        _fft_convolve_same_length(signal, delta), signal, atol=1e-6, rtol=0
    )


def test_robust_modes_with_user_rirs(backend, audio, patch_psycho):
    rooms = RoomSimulator.from_rirs(
        [
            np.array([1.0], dtype=np.float64),
            np.array([0.9, 0.1], dtype=np.float64),
            np.array([0.8, 0.15, 0.05], dtype=np.float64),
        ]
    )
    with pytest.raises(ValueError, match="rooms"):
        imperceptible(
            backend,
            audio,
            "A",
            mode="robust",
            num_iter1=1,
            num_iter2=1,
            nested=False,
            verbose=False,
        )

    robust = imperceptible(
        backend,
        audio,
        "A",
        mode="robust",
        rooms=rooms,
        epsilon=0.05,
        num_iter_r1=2,
        num_iter_r2=2,
        check_every=1,
        nested=False,
        verbose=False,
    )
    assert robust.shape == audio.numpy().shape
    assert np.isfinite(robust).all()

    ir = imperceptible(
        backend,
        audio,
        "A",
        mode="imperceptible_robust",
        rooms=rooms,
        epsilon=0.05,
        num_iter_r1=1,
        num_iter_r2=1,
        num_iter_ir1=2,
        num_iter_ir2=2,
        check_every=1,
        nested=False,
        verbose=False,
        sample_rate=16000,
    )
    assert ir.shape == audio.numpy().shape
    assert np.isfinite(ir).all()


def test_untargeted_imperceptible_accepts_label(backend, audio, patch_psycho):
    out = imperceptible(
        backend,
        audio,
        targeted=False,
        label="CAB",
        epsilon=0.05,
        num_iter1=3,
        num_iter2=3,
        early_stop_cw=False,
        search_eps_cw=False,
        nested=False,
        verbose=False,
        sample_rate=16000,
    )
    assert out.shape == audio.numpy().shape
    assert np.isfinite(out).all()


def test_attacker_wrappers_accept_new_kwargs(backend, audio, patch_psycho):
    """ASRAttacker.cw / .imperceptible forward label / loss / mode without TypeError."""
    from asr_attacks.attacker import ASRAttacker
    from asr_attacks.compat import ASRAttacks
    from tests.helpers import LABELS, TinyCTC

    attacker = ASRAttacker(backend, verbose=False)
    out = attacker.cw(
        audio,
        label="CAB",
        targeted=False,
        num_iter=2,
        early_stop=False,
        search_eps=False,
        nested=False,
        loss="ctc",
        db_bound=None,
        kappa=0.0,
        check_every=1,
    )
    assert isinstance(out, np.ndarray)

    with pytest.raises(ValueError, match="targeted"):
        attacker.cw(
            audio,
            loss="margin",
            targeted=False,
            num_iter=1,
            early_stop=False,
            search_eps=False,
        )

    out = attacker.imperceptible(
        audio,
        targeted=False,
        label="CAB",
        mode="imperceptible",
        num_iter1=2,
        num_iter2=2,
        early_stop_cw=False,
        search_eps_cw=False,
        nested=False,
        sample_rate=16000,
    )
    assert isinstance(out, np.ndarray)

    compat = ASRAttacks(TinyCTC(), "cpu", LABELS)
    out = compat.CW_ATTACK(
        audio,
        label="CAB",
        targeted=False,
        num_iter=2,
        early_stop=False,
        search_eps=False,
        nested=False,
        loss="ctc",
        check_every=1,
    )
    assert isinstance(out, np.ndarray)
    out = compat.IMPERCEPTIBLE_ATTACK(
        audio,
        targeted=False,
        label="CAB",
        mode="imperceptible",
        num_iter1=2,
        num_iter2=2,
        early_stop_cw=False,
        search_eps_cw=False,
        nested=False,
        sample_rate=16000,
    )
    assert isinstance(out, np.ndarray)

    with pytest.raises(ValueError, match="target"):
        compat.IMPERCEPTIBLE_ATTACK(audio, targeted=True, num_iter1=1, num_iter2=1, nested=False)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyroomacoustics") is None,
    reason="pyroomacoustics not installed",
)
def test_room_simulator_pyroomacoustics_generates():
    rooms = RoomSimulator(num_rooms=2, seed=0, sample_rate=8000)
    assert len(rooms.rirs) == 2
    x = torch.randn(200)
    y = rooms.sample()(x)
    assert y.shape == x.shape
