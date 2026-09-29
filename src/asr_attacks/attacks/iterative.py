from __future__ import annotations

import math

import numpy as np
import torch

from asr_attacks.attacks.common import (
    ctc_loss,
    iteration_bar,
    maybe_early_stop,
    resolve_target,
)
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.tensors import (
    _parse_norm,
    normalized_step,
    prepare_audio,
    project,
    random_ball,
    to_numpy,
)


def bim_num_iter(epsilon: float, alpha: float) -> int:
    """Kurakin et al. 2017 iteration count: ``ceil(min(eps/alpha + 4, 1.25 * eps/alpha))``."""
    ratio = epsilon / alpha
    return int(math.ceil(min(ratio + 4, 1.25 * ratio)))


def _iterative(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float,
    alpha: float,
    num_iter: int,
    norm: float | str = "inf",
    targeted: bool = False,
    target: str | list[str] | None = None,
    label: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    random_start: bool = False,
    verbose: bool = True,
    desc: str | None = None,
    target_text: str | None = None,
) -> np.ndarray:
    _parse_norm(norm)
    original = prepare_audio(audio, backend.device).detach()
    if target_text is None:
        target_text = resolve_target(backend, original, target, targeted, label)
    target_ids = backend.encode(target_text)
    sign = -1.0 if targeted else 1.0

    if random_start:
        adversarial = project(
            original + random_ball(original, epsilon, norm),
            original,
            epsilon,
            norm,
        )
    else:
        adversarial = original.clone()

    success_message = (
        "Stopping early: targeted transcription reached."
        if targeted
        else "Stopping early: untargeted attack changed the transcription."
    )

    for _ in iteration_bar(num_iter, nested=nested, desc=desc, verbose=verbose):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = backend.logits(adversarial)
        loss = ctc_loss(logits, target_ids, backend.blank_id)
        loss.backward()
        if adversarial.grad is None:
            raise RuntimeError("Iterative attack did not receive input gradients")
        stepped = adversarial.detach() + alpha * sign * normalized_step(adversarial.grad, norm)
        adversarial = project(stepped, original, epsilon, norm)
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
    alpha: float | None = None,
    num_iter: int | None = None,
    targeted: bool = False,
    target: str | list[str] | None = None,
    label: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    verbose: bool = True,
    norm: float | str = "inf",
) -> np.ndarray:
    """Basic Iterative Method (Kurakin et al., 2017).

    Paper: https://arxiv.org/abs/1607.02533

    ``x_{n+1} = Clip_{X,eps}(x_n + alpha * s * normalize_p(grad CTC(f(x_n), y)))``
    with ``s = +1`` untargeted and ``s = -1`` targeted. ``Clip_{X,eps}`` is the
    L-``norm`` ball projection around the clean waveform followed by clamping to
    ``[-1, 1]``. ``normalize_p`` is ``sign(g)`` for ``norm="inf"``,
    ``g / ||g||_2`` for ``norm=2``, and ``g / ||g||_1`` for ``norm=1``.

    Defaults: ``alpha = epsilon / 10``; ``num_iter`` follows the paper rule
    ``ceil(min(eps/alpha + 4, 1.25 * eps/alpha))``. The paper is L-inf only;
    ``norm=2`` / ``norm=1`` are package extensions ("L2 / L1 BIM"). Targeted
    mode with a user-chosen transcript covers the paper's least-likely-class
    idea (ASR has no natural least-likely transcript). Early stopping is a
    package extension (not in the paper or ART BIM).
    """
    if alpha is None:
        alpha = epsilon / 10.0
    if num_iter is None:
        num_iter = bim_num_iter(epsilon, alpha)
    return _iterative(
        backend,
        audio,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        norm=norm,
        targeted=targeted,
        target=target,
        label=label,
        nested=nested,
        early_stop=early_stop,
        random_start=False,
        verbose=verbose,
        desc=None,
    )


def _ctc_value(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    target_ids: torch.Tensor,
) -> float:
    tensor = prepare_audio(audio, backend.device)
    with torch.no_grad():
        return float(ctc_loss(backend.logits(tensor), target_ids, backend.blank_id))


def _prefer_restart(
    *,
    success: bool,
    loss: float,
    best_success: bool | None,
    best_loss: float | None,
    targeted: bool,
) -> bool:
    if best_success is None:
        return True
    if success and not best_success:
        return True
    if success == best_success:
        return loss < best_loss if targeted else loss > best_loss
    return False


def pgd(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.3,
    alpha: float | None = None,
    num_iter: int = 40,
    targeted: bool = False,
    target: str | list[str] | None = None,
    label: str | list[str] | None = None,
    nested: bool = True,
    early_stop: bool = False,
    random_start: bool = True,
    verbose: bool = True,
    norm: float | str = "inf",
    restarts: int = 1,
) -> np.ndarray:
    """Projected Gradient Descent (Madry et al., 2018).

    Paper: https://arxiv.org/abs/1706.06083

    ``x_{n+1} = Proj_{X,eps}(x_n + alpha * s * normalize_p(grad CTC(f(x_n), y)))``
    after an optional random start inside the L-``norm`` ball. ``s = +1``
    untargeted, ``s = -1`` targeted. Default ``alpha = 2.5 * epsilon / num_iter``.

    L-inf is the main Madry attack; L2 (normalized gradient steps) is cited to
    Sec. 5 of the paper. L1 follows the ART convention and is a package
    extension. Early stopping is also a package extension. PGD with the CW
    margin loss is not provided (no direct CTC equivalent).

    ``restarts`` runs the loop from a fresh random start each time. The kept
    result is any successful restart (``attack_succeeded`` vs the clean-audio
    reference), else the restart with the strongest final CTC loss (highest
    untargeted, lowest targeted). ``restarts > 1`` requires ``random_start``.
    With ``early_stop=True``, the first successful restart is returned immediately.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1; got {restarts}")
    if restarts > 1 and not random_start:
        raise ValueError("restarts > 1 requires random_start=True")
    if alpha is None:
        alpha = 2.5 * epsilon / num_iter

    original = prepare_audio(audio, backend.device).detach()
    target_text = resolve_target(backend, original, target, targeted, label)
    target_ids = backend.encode(target_text)

    best: np.ndarray | None = None
    best_success: bool | None = None
    best_loss: float | None = None

    for _ in range(restarts):
        candidate = _iterative(
            backend,
            audio,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            norm=norm,
            targeted=targeted,
            nested=nested,
            early_stop=early_stop,
            random_start=random_start,
            verbose=verbose,
            desc=None,
            target_text=target_text,
        )
        hypothesis = backend.decode(prepare_audio(candidate, backend.device))
        success = attack_succeeded(hypothesis, target_text, targeted=targeted)
        if early_stop and success:
            return candidate
        loss = _ctc_value(backend, candidate, target_ids)
        if _prefer_restart(
            success=success,
            loss=loss,
            best_success=best_success,
            best_loss=best_loss,
            targeted=targeted,
        ):
            best = candidate
            best_success = success
            best_loss = loss

    assert best is not None
    return best
