from __future__ import annotations

import pytest
import torch

from asr_attacks.backends.module import CTCModuleBackend
from tests.helpers import LABELS, TinyCTC


@pytest.fixture
def backend() -> CTCModuleBackend:
    torch.manual_seed(0)
    return CTCModuleBackend(TinyCTC(), labels=LABELS, device="cpu")


@pytest.fixture
def audio() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 1600) * 0.05
