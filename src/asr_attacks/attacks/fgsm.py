from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacks.common import ctc_loss, resolve_target
from asr_attacks.backends.base import ASRBackend
from asr_attacks.tensors import _parse_norm, normalized_step, prepare_audio, to_numpy


def fgsm(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.2,
    targeted: bool = False,
    target: str | list[str] | None = None,
    label: str | list[str] | None = None,
    norm: float | str = "inf",
) -> np.ndarray:
    """Fast Gradient Sign / Fast Gradient Method, one step.

    ``x_adv = clip_[-1, 1](x + epsilon * s * normalize_p(grad_x CTC(f(x), y)))``
    with ``s = +1`` untargeted (``y`` = ``label`` or the clean greedy decode) and
    ``s = -1`` targeted (``y`` = ``target``). ``normalize_p`` is ``sign(g)`` for
    ``norm="inf"``, ``g / ||g||_2`` for ``norm=2`` and ``g / ||g||_1`` for
    ``norm=1`` (the ART convention).

    - L-inf is FGSM (Goodfellow et al., 2015, https://arxiv.org/abs/1412.6572).
    - L2 / L1 are the Fast Gradient Method, and targeted FGSM follows Kurakin
      et al., 2017 (https://arxiv.org/abs/1611.01236).

    ``label`` is the ground-truth transcript for untargeted attacks (the
    Goodfellow et al. setup). By default the model's clean prediction is used,
    as in Kurakin et al. 2017 and ART. Passing ``label`` with ``targeted=True``
    raises ``ValueError``.
    """
    _parse_norm(norm)
    original = prepare_audio(audio, backend.device).detach()
    adversarial = original.clone().requires_grad_(True)
    target_text = resolve_target(backend, original, target, targeted, label)
    target_ids = backend.encode(target_text)
    logits = backend.logits(adversarial)
    loss = ctc_loss(logits, target_ids, backend.blank_id)
    loss.backward()
    if adversarial.grad is None:
        raise RuntimeError(
            "FGSM did not receive input gradients; check that the model is differentiable"
        )
    sign = -1.0 if targeted else 1.0
    step = epsilon * sign * normalized_step(adversarial.grad, norm)
    perturbed = torch.clamp(adversarial.detach() + step, -1.0, 1.0)
    return to_numpy(perturbed)
