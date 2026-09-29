from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from asr_attacks.attacker import ASRAttacker
from asr_attacks.attacks.common import ctc_loss
from asr_attacks.attacks.iterative import bim, bim_num_iter, pgd
from asr_attacks.compat import ASRAttacks
from asr_attacks.tensors import _parse_norm, prepare_audio, project, random_ball
from tests.helpers import LABELS

NORMS = ["inf", 2, 1]


def _lp(delta: np.ndarray | torch.Tensor, norm) -> float:
    p = _parse_norm(norm)
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    if p == float("inf"):
        return float(np.abs(values).max())
    return float(np.linalg.norm(values, ord=p))


def _ctc(backend, waveform: np.ndarray, text: str) -> float:
    tensor = prepare_audio(waveform, backend.device)
    with torch.no_grad():
        return float(ctc_loss(backend.logits(tensor), backend.encode(text), backend.blank_id))


@pytest.mark.parametrize("norm", NORMS)
def test_bim_respects_norm_ball_and_range(backend, audio, norm):
    epsilon = 0.05
    adversarial = bim(
        backend,
        audio,
        epsilon=epsilon,
        alpha=0.01,
        num_iter=8,
        norm=norm,
        nested=False,
        verbose=False,
    )
    assert _lp(adversarial - audio.numpy(), norm) <= epsilon + 1e-5
    assert adversarial.max() <= 1.0 + 1e-5
    assert adversarial.min() >= -1.0 - 1e-5


def test_bim_num_iter_paper_rule():
    assert bim_num_iter(0.2, 0.02) == 13
    assert bim_num_iter(0.1, 0.01) == 13
    assert bim_num_iter(1.0, 1.0) == 2  # min(5, 1.25) -> ceil 2
    ratio = 0.3 / 0.05
    assert bim_num_iter(0.3, 0.05) == int(math.ceil(min(ratio + 4, 1.25 * ratio)))


def test_bim_num_iter_none_uses_paper_count(backend, audio, monkeypatch):
    seen: list[int] = []

    def fake_bar(num_iter, nested, desc=None):
        seen.append(num_iter)
        return range(num_iter)

    monkeypatch.setattr("asr_attacks.attacks.iterative.iteration_bar", fake_bar)
    epsilon, alpha = 0.2, 0.02
    bim(
        backend,
        audio,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=None,
        nested=False,
        verbose=False,
    )
    assert seen == [bim_num_iter(epsilon, alpha)]


def test_bim_alpha_none_defaults_to_epsilon_over_ten(backend, audio, monkeypatch):
    seen: list[int] = []

    def fake_bar(num_iter, nested, desc=None):
        seen.append(num_iter)
        return range(num_iter)

    monkeypatch.setattr("asr_attacks.attacks.iterative.iteration_bar", fake_bar)
    epsilon = 0.2
    bim(backend, audio, epsilon=epsilon, alpha=None, num_iter=None, nested=False, verbose=False)
    assert seen == [bim_num_iter(epsilon, epsilon / 10.0)]


@pytest.mark.parametrize("norm", NORMS)
def test_pgd_random_start_and_final_in_ball(backend, audio, norm, monkeypatch):
    epsilon = 0.04
    starts: list[torch.Tensor] = []
    real_project = project

    def capturing_project(adversarial, original, eps, n, clip_min=-1.0, clip_max=1.0):
        out = real_project(adversarial, original, eps, n, clip_min=clip_min, clip_max=clip_max)
        if not starts:
            starts.append(out.detach().clone())
        return out

    monkeypatch.setattr("asr_attacks.attacks.iterative.project", capturing_project)
    torch.manual_seed(0)
    adversarial = pgd(
        backend,
        audio,
        epsilon=epsilon,
        alpha=0.01,
        num_iter=6,
        norm=norm,
        nested=False,
        verbose=False,
        random_start=True,
        restarts=1,
    )
    assert starts, "expected a random-start projection"
    start = starts[0]
    assert _lp(start - audio, norm) <= epsilon + 1e-5
    assert start.max() <= 1.0 + 1e-5 and start.min() >= -1.0 - 1e-5
    assert _lp(adversarial - audio.numpy(), norm) <= epsilon + 1e-5
    assert adversarial.max() <= 1.0 + 1e-5
    assert adversarial.min() >= -1.0 - 1e-5


def test_pgd_restarts_at_least_as_strong(backend, audio):
    kwargs = dict(
        epsilon=0.05,
        alpha=0.02,
        num_iter=5,
        nested=False,
        verbose=False,
        random_start=True,
        early_stop=False,
    )
    torch.manual_seed(7)
    one = pgd(backend, audio, restarts=1, **kwargs)
    torch.manual_seed(7)
    three = pgd(backend, audio, restarts=3, **kwargs)
    reference = backend.decode(audio)
    assert _ctc(backend, three, reference) >= _ctc(backend, one, reference) - 1e-6


def test_pgd_restarts_require_random_start(backend, audio):
    with pytest.raises(ValueError, match="random_start"):
        pgd(
            backend,
            audio,
            epsilon=0.03,
            alpha=0.01,
            num_iter=2,
            restarts=2,
            random_start=False,
            nested=False,
            verbose=False,
        )


def test_pgd_random_ball_start_matches_helpers(audio):
    torch.manual_seed(3)
    original = prepare_audio(audio, "cpu")
    epsilon = 0.08
    for norm in NORMS:
        torch.manual_seed(3 + hash(str(norm)) % 1000)
        delta = random_ball(original, epsilon, norm)
        started = project(original + delta, original, epsilon, norm)
        assert _lp(started - original, norm) <= epsilon + 1e-5
        assert started.max() <= 1.0 and started.min() >= -1.0


def test_wrappers_pass_norm_alpha_restarts(backend, audio):
    attacker = ASRAttacker(backend, verbose=False)
    direct_bim = bim(
        backend, audio, epsilon=0.05, alpha=None, num_iter=None, norm=2, nested=False, verbose=False
    )
    np.testing.assert_allclose(
        attacker.bim(audio, epsilon=0.05, alpha=None, num_iter=None, norm=2, nested=False),
        direct_bim,
    )

    torch.manual_seed(0)
    direct_pgd = pgd(
        backend,
        audio,
        epsilon=0.05,
        alpha=None,
        num_iter=4,
        norm=2,
        nested=False,
        verbose=False,
        restarts=1,
    )
    torch.manual_seed(0)
    np.testing.assert_allclose(
        attacker.pgd(audio, epsilon=0.05, alpha=None, num_iter=4, norm=2, nested=False, restarts=1),
        direct_pgd,
    )

    compat = ASRAttacks(backend.model, "cpu", LABELS)
    np.testing.assert_allclose(
        compat.BIM_ATTACK(audio, epsilon=0.05, alpha=None, num_iter=None, norm=2, nested=False),
        direct_bim,
        atol=1e-7,
    )
    torch.manual_seed(0)
    np.testing.assert_allclose(
        compat.PGD_ATTACK(
            audio, epsilon=0.05, alpha=None, num_iter=4, norm=2, nested=False, restarts=1
        ),
        direct_pgd,
        atol=1e-7,
    )
