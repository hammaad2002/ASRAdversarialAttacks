from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacks.common import (
    ctc_loss,
    iteration_bar,
    maybe_early_stop,
    resolve_target,
)
from asr_attacks.backends.base import ASRBackend
from asr_attacks.tensors import prepare_audio, project_linf, to_numpy


def linf_iterative(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float,
    alpha: float,
    num_iter: int,
    targeted: bool = False,
    target: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    random_start: bool = False,
    verbose: bool = True,
    desc: str | None = None,
) -> np.ndarray:
    original = prepare_audio(audio, backend.device).detach()
    target_text = resolve_target(backend, original, target, targeted)
    target_ids = backend.encode(target_text)
    sign = -1.0 if targeted else 1.0

    if random_start:
        noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
        adversarial = project_linf(original + noise, original, epsilon)
    else:
        adversarial = original.clone()

    success_message = (
        "Stopping early: targeted transcription reached."
        if targeted
        else "Stopping early: untargeted attack changed the transcription."
    )

    for _ in iteration_bar(num_iter, nested=nested, desc=desc):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = backend.logits(adversarial)
        loss = ctc_loss(logits, target_ids, backend.blank_id)
        loss.backward()
        if adversarial.grad is None:
            raise RuntimeError("Iterative attack did not receive input gradients")
        stepped = adversarial.detach() + alpha * sign * adversarial.grad.sign()
        adversarial = project_linf(stepped, original, epsilon)
        if maybe_early_stop(
            backend,
            adversarial,
            target_text,
            targeted=targeted,
            early_stop=early_stop,
            verbose=verbose,
            success_message=success_message,
        ):
            break
    return to_numpy(adversarial)


def bim(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.2,
    alpha: float = 0.1,
    num_iter: int = 10,
    targeted: bool = False,
    target: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Basic Iterative Method (Kurakin et al., 2017).

    Paper: https://arxiv.org/abs/1607.02533

    Each step is projected onto the L-inf ball of radius ``epsilon`` around the
    original waveform, then clipped to ``[-1, 1]``.
    """
    return linf_iterative(
        backend,
        audio,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        targeted=targeted,
        target=target,
        nested=nested,
        early_stop=early_stop,
        random_start=False,
        verbose=verbose,
        desc=None,
    )


def pgd(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.3,
    alpha: float = 0.01,
    num_iter: int = 40,
    targeted: bool = False,
    target: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    random_start: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Projected Gradient Descent (Madry et al., 2018).

    Paper: https://arxiv.org/abs/1706.06083

    Defaults to a random start inside the L-inf ball. Set ``random_start=False``
    to recover BIM-style initialization.
    """
    return linf_iterative(
        backend,
        audio,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        targeted=targeted,
        target=target,
        nested=nested,
        early_stop=early_stop,
        random_start=random_start,
        verbose=verbose,
        desc=None,
    )
