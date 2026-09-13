from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ASRBackend(ABC):
    """CTC model wrapper used by every attack."""

    device: torch.device
    blank_id: int

    @abstractmethod
    def logits(self, audio: torch.Tensor) -> torch.Tensor:
        """Return log-unnormalized CTC logits of shape ``(batch, time, vocab)``."""

    @abstractmethod
    def encode(self, transcript: str) -> torch.Tensor:
        """Return 1-D token ids for a CTC target sequence."""

    @abstractmethod
    def decode(self, audio: torch.Tensor) -> str:
        """Greedy CTC decode of ``audio`` without tracking gradients."""
