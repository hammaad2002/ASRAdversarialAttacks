"""``verbose=False`` keeps every attack completely silent (no tqdm bars, no prints)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from asr_attacks.attacker import ASRAttacker
from asr_attacks.attacks.common import iteration_bar
from asr_attacks.attacks.cw import cw
from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.imperceptible import imperceptible
from asr_attacks.attacks.iterative import bim, pgd
from asr_attacks.compat import ASRAttacks
from asr_attacks.rooms import RoomSimulator
from tests.helpers import LABELS, TinyCTC

# Long enough for the real 2048-sample masking window.
LONG = 3 * 2048


@pytest.fixture
def long_audio() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(1, LONG) * 0.05


@pytest.fixture
def rooms() -> RoomSimulator:
    return RoomSimulator.from_rirs(
        [np.array([1.0]), np.array([0.9, 0.1]), np.array([0.8, 0.15, 0.05])]
    )


def _run_every_attack(backend, audio, long_audio, rooms, verbose):
    quiet = {"nested": False, "verbose": verbose}
    fgsm(backend, audio, epsilon=0.01, targeted=True, target="AB")
    bim(backend, audio, epsilon=0.01, num_iter=2, targeted=True, target="AB", **quiet)
    pgd(backend, audio, epsilon=0.01, num_iter=2, restarts=2, **quiet)
    cw_kwargs = dict(num_iter=2, early_stop=False, search_eps=False, targeted=True, target="AB")
    cw(backend, audio, **cw_kwargs, **quiet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cw(backend, audio, loss="margin", **cw_kwargs, **quiet)
    imperceptible(
        backend,
        long_audio,
        "AB",
        num_iter1=2,
        num_iter2=2,
        early_stop_cw=False,
        search_eps_cw=False,
        **quiet,
    )
    imperceptible(
        backend,
        long_audio,
        "AB",
        mode="imperceptible_robust",
        rooms=rooms,
        num_iter_r1=2,
        num_iter_r2=2,
        num_iter_ir1=2,
        num_iter_ir2=2,
        check_every=1,
        **quiet,
    )


def test_verbose_false_prints_nothing(backend, audio, long_audio, rooms, capsys):
    _run_every_attack(backend, audio, long_audio, rooms, verbose=False)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_verbose_true_draws_progress_bars(backend, audio, long_audio, rooms, capsys):
    """Positive control: the silence above is caused by ``verbose``, not by capture."""
    _run_every_attack(backend, audio, long_audio, rooms, verbose=True)
    captured = capsys.readouterr()
    for stage in ("*****Attack Stage 1*****", "*****Attack Stage 2*****", "*****Robust R1*****"):
        assert stage in captured.err


def test_iteration_bar_respects_verbose(capsys):
    assert list(iteration_bar(3, nested=False, verbose=False)) == [0, 1, 2]
    assert capsys.readouterr().err == ""
    assert list(iteration_bar(3, nested=False, desc="loop")) == [0, 1, 2]
    assert "loop" in capsys.readouterr().err


def test_attacker_verbose_flag_reaches_iterative_attacks(backend, audio, capsys):
    ASRAttacker(backend, verbose=False).bim(audio, epsilon=0.01, num_iter=2, nested=False)
    assert capsys.readouterr().err == ""
    ASRAttacker(backend, verbose=True).bim(audio, epsilon=0.01, num_iter=2, nested=False)
    assert capsys.readouterr().err != ""


def test_compat_wrapper_can_be_silenced(audio, capsys):
    attacks = ASRAttacks(TinyCTC(), "cpu", LABELS, verbose=False)
    attacks.BIM_ATTACK(audio, epsilon=0.01, num_iter=2, nested=False)
    attacks.PGD_ATTACK(audio, epsilon=0.01, num_iter=2, nested=False)
    assert capsys.readouterr().err == ""
