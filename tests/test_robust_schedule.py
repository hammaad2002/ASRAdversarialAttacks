"""Robust stages only decode the M rooms on their scheduled check steps.

Decoding ``M`` rooms is far more expensive than it looks on a toy model, so the number
of ``backend.decode`` calls is pinned exactly. The stub decode either never matches the
target ("attack never succeeds") or always does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_attacks.attacks.imperceptible import imperceptible
from asr_attacks.rooms import RoomSimulator

M_ROOMS = 10  # fixed by the robust stages (Qin et al. sample M = 10 rooms per step)


@pytest.fixture
def rooms() -> RoomSimulator:
    return RoomSimulator.from_rirs(
        [np.array([1.0]), np.array([0.9, 0.1]), np.array([0.8, 0.15, 0.05])]
    )


@pytest.fixture
def long_audio() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(1, 3 * 2048) * 0.05  # long enough for the real masking window


class DecodeCounter:
    """Replace ``backend.decode`` with a counter that returns a fixed transcript."""

    def __init__(self, backend, transcript: str) -> None:
        self.calls = 0
        self.transcript = transcript
        backend.decode = self  # type: ignore[method-assign]

    def __call__(self, audio) -> str:
        self.calls += 1
        return self.transcript


def _robust(backend, audio, rooms, **kwargs):
    defaults = {
        "mode": "robust",
        "rooms": rooms,
        "epsilon": 0.05,
        "nested": False,
        "verbose": False,
    }
    return imperceptible(backend, audio, "AB", **{**defaults, **kwargs})


def test_r1_and_r2_check_only_every_check_every_steps(backend, audio, rooms):
    counter = DecodeCounter(backend, transcript="")  # never reaches the target

    _robust(backend, audio, rooms, num_iter_r1=20, num_iter_r2=30, check_every=10)

    # R1: checks at steps 10 and 20, one room each. R2: checks at 10, 20 and 30, and the
    # "all M rooms" test stops at the first failing room. Decoding per step would be 50.
    assert counter.calls == 2 + 3


def test_check_every_one_restores_per_step_checks(backend, audio, rooms):
    counter = DecodeCounter(backend, transcript="")
    _robust(backend, audio, rooms, num_iter_r1=20, num_iter_r2=30, check_every=1)
    assert counter.calls == 20 + 30


def test_r2_decodes_all_m_rooms_when_the_attack_succeeds(backend, audio, rooms):
    counter = DecodeCounter(backend, transcript="AB")  # matches the target everywhere

    out = _robust(backend, audio, rooms, num_iter_r1=20, num_iter_r2=30, check_every=10)

    # R1: 2 checks x 1 room. R2: 3 checks x all M rooms, because success needs every room.
    assert counter.calls == 2 + 3 * M_ROOMS
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    ("check_every", "ir1_checks", "ir2_checks"),
    [
        # IR1 has 20 iterations, IR2 has 60. Failure boundaries at step 50 also decode.
        (10, [10, 20], [10, 20, 30, 40, 50, 60]),
        (20, [20], [20, 40, 50, 60]),
    ],
)
def test_ir_loops_decode_m_rooms_only_on_scheduled_steps(
    backend, long_audio, rooms, check_every, ir1_checks, ir2_checks
):
    counter = DecodeCounter(backend, transcript="")

    imperceptible(
        backend,
        long_audio,
        "AB",
        mode="imperceptible_robust",
        rooms=rooms,
        epsilon=0.05,
        num_iter_r1=1,
        num_iter_r2=1,
        num_iter_ir1=20,
        num_iter_ir2=60,
        check_every=check_every,
        nested=False,
        verbose=False,
    )

    # R1/R2 run one step each, so they never reach a check. Every IR check decodes all M
    # rooms; per-step decoding would have cost 80 * M = 800.
    assert counter.calls == (len(ir1_checks) + len(ir2_checks)) * M_ROOMS


def test_ir_success_keeps_the_penalty_schedule_and_returns_valid_audio(backend, long_audio, rooms):
    counter = DecodeCounter(backend, transcript="AB")

    out = imperceptible(
        backend,
        long_audio,
        "AB",
        mode="imperceptible_robust",
        rooms=rooms,
        epsilon=0.05,
        num_iter_r1=1,
        num_iter_r2=1,
        num_iter_ir1=20,
        num_iter_ir2=20,
        check_every=10,
        nested=False,
        verbose=False,
    )

    assert counter.calls == (2 + 2) * M_ROOMS
    assert out.shape == long_audio.numpy().shape
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0 + 1e-5
