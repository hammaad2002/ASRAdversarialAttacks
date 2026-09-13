from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacks.common import ctc_loss, resolve_target
from asr_attacks.backends.base import ASRBackend
from asr_attacks.tensors import prepare_audio, to_numpy


def fgsm(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.2,
    targeted: bool = False,
    target: str | list[str] | None = None,
) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Paper: https://arxiv.org/abs/1412.6572
    """
    original = prepare_audio(audio, backend.device).detach()
    adversarial = original.clone().requires_grad_(True)
    target_text = resolve_target(backend, original, target, targeted)
    target_ids = backend.encode(target_text)
    logits = backend.logits(adversarial)
    loss = ctc_loss(logits, target_ids, backend.blank_id)
    loss.backward()
    if adversarial.grad is None:
        raise RuntimeError(
            "FGSM did not receive input gradients; check that the model is differentiable"
        )
    signed = -adversarial.grad.sign() if targeted else adversarial.grad.sign()
    perturbed = torch.clamp(adversarial.detach() + epsilon * signed, -1.0, 1.0)
    return to_numpy(perturbed)
