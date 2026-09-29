from __future__ import annotations

import pytest
import torch

from asr_attacks.backends.module import CTCModuleBackend
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
def audio() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 1600) * 0.05
