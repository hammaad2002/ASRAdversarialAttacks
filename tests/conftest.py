from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_attacks.backends.module import CTCModuleBackend
from asr_attacks.rooms import RoomSimulator
from tests.helpers import LABELS, TinyCTC


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_torch():
    """The fixtures use toy tensors, where torch's thread pool costs ~10x more than it saves."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def backend() -> CTCModuleBackend:
    torch.manual_seed(0)
    return CTCModuleBackend(TinyCTC(), labels=LABELS, device="cpu")


@pytest.fixture
def silent_backend() -> CTCModuleBackend:
    """A model whose clean transcript is empty, so a silence target is already reached."""
    torch.manual_seed(0)
    model = TinyCTC()
    with torch.no_grad():
        model.conv.weight.mul_(0.1)
        model.conv.bias.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]))
    return CTCModuleBackend(model, labels=LABELS, device="cpu")


@pytest.fixture
def audio() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 1600) * 0.05


@pytest.fixture
def long_audio() -> torch.Tensor:
    """Three masking windows (2048 samples each), so the real psychoacoustic code can run."""
    torch.manual_seed(2)
    return torch.randn(1, 3 * 2048) * 0.05


@pytest.fixture
def rooms() -> RoomSimulator:
    """Three user-supplied impulse responses: no pyroomacoustics needed."""
    return RoomSimulator.from_rirs(
        [np.array([1.0]), np.array([0.9, 0.1]), np.array([0.8, 0.15, 0.05])]
    )
