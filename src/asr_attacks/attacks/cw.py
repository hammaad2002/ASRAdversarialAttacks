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
from asr_attacks.metrics import attack_succeeded
from asr_attacks.psychoacoustic import compute_masking_threshold, psd_transform
from asr_attacks.tensors import prepare_audio, to_numpy
from asr_attacks.text import as_transcript


def _optimizer_for(name: str | None, parameters, learning_rate: float) -> torch.optim.Optimizer:
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    return torch.optim.SGD(parameters, lr=learning_rate)


def cw(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 0.3,
    c: float = 1e-4,
    learning_rate: float = 0.01,
    num_iter: int = 1000,
    decrease_factor_eps: float = 1.0,
    num_iter_decrease_eps: int = 10,
    optimizer: str | None = None,
    nested: bool = True,
    early_stop: bool = True,
    search_eps: bool = False,
    targeted: bool = False,
    target: str | list[str] | None = None,
    verbose: bool = True,
    desc: str | None = None,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """CTC + L2 optimization inside an L-inf box.

    Inspired by Carlini & Wagner (https://arxiv.org/abs/1801.01944) but this is
    **not** a paper-faithful C&W implementation: there is no tanh change of
    variables and no binary search on ``c``. The objective is
    ``c * ||delta||_2 + CTC`` (sign flipped for untargeted).
    """
    if early_stop and search_eps:
        raise ValueError("early_stop and search_eps cannot both be True")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")

    original = prepare_audio(audio, backend.device).detach()
    adversarial = original.clone().requires_grad_(True)
    opt = _optimizer_for(optimizer, [adversarial], learning_rate)
    target_text = resolve_target(backend, original, target, targeted)
    target_ids = backend.encode(target_text)
    sign = 1.0 if targeted else -1.0
    successful_streak = 0

    success_message = (
        "Stopping early: targeted transcription reached."
        if targeted
        else "Stopping early: untargeted attack changed the transcription."
    )

    for _ in iteration_bar(num_iter, nested=nested, desc=desc):
        opt.zero_grad()
        logits = backend.logits(adversarial)
        loss_classifier = sign * ctc_loss(logits, target_ids, backend.blank_id)
        loss = (c * torch.norm(adversarial - original)) + loss_classifier
        loss.backward()
        opt.step()
        perturbation = torch.clamp(adversarial.detach() - original, -epsilon, epsilon)
        adversarial.data = torch.clamp(original + perturbation, -1.0, 1.0)

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

        if search_eps:
            if attack_succeeded(backend.decode(adversarial), target_text, targeted=targeted):
                successful_streak += 1
                if successful_streak >= num_iter_decrease_eps:
                    epsilon *= decrease_factor_eps
                    successful_streak = 0

    if return_tensor:
        return adversarial.detach()
    return to_numpy(adversarial)


def imperceptible(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    target: str | list[str],
    *,
    epsilon: float = 0.3,
    c: float = 1e-4,
    learning_rate1: float = 0.01,
    learning_rate2: float = 0.01,
    num_iter1: int = 10000,
    num_iter2: int = 2000,
    decrease_factor_eps: float = 1.0,
    num_iter_decrease_eps: int = 10,
    optimizer1: str | None = None,
    optimizer2: str | None = None,
    nested: bool = True,
    early_stop_cw: bool = True,
    search_eps_cw: bool = False,
    alpha: float = 0.5,
    sample_rate: int = 16000,
    verbose: bool = True,
) -> np.ndarray:
    """Two-stage imperceptible ASR attack (Qin et al., 2019).

    Paper: https://arxiv.org/abs/1903.10346

    Stage 1 is the local CW-style optimizer. Stage 2 penalizes perturbation
    energy above a psychoacoustic masking threshold (ART-derived).
    """
    original = prepare_audio(audio, backend.device).detach()
    stage_one = cw(
        backend,
        original,
        epsilon=epsilon,
        c=c,
        learning_rate=learning_rate1,
        num_iter=num_iter1,
        decrease_factor_eps=decrease_factor_eps,
        num_iter_decrease_eps=num_iter_decrease_eps,
        optimizer=optimizer1,
        nested=True,
        early_stop=early_stop_cw,
        search_eps=search_eps_cw,
        targeted=True,
        target=target,
        verbose=verbose,
        desc="*****Attack Stage 1*****",
        return_tensor=True,
    )
    assert isinstance(stage_one, torch.Tensor)

    adversarial = stage_one.clone().detach().requires_grad_(True)
    opt = _optimizer_for(optimizer2, [adversarial], learning_rate2)
    target_ids = backend.encode(target)
    waveform = to_numpy(original).squeeze()
    theta, original_max_psd = compute_masking_threshold(waveform, sample_rate=sample_rate)
    theta_t = torch.tensor(theta.transpose(1, 0), device=backend.device)
    relu = torch.nn.ReLU()
    alpha_t = alpha
    buffer_losses: list[torch.Tensor] = []
    buffer_examples: list[np.ndarray] = []

    for step in iteration_bar(num_iter2, nested=nested, desc="*****Attack Stage 2*****"):
        opt.zero_grad()
        logits = backend.logits(adversarial)
        loss_classifier = ctc_loss(logits, target_ids, backend.blank_id)
        loss_regularizer = torch.norm(adversarial - original)
        loss1 = (torch.tensor(c, device=backend.device) * loss_classifier) + loss_regularizer
        perturbation = adversarial - original
        psd_delta = psd_transform(perturbation, original_max_psd, backend.device)
        loss2 = torch.mean(relu(psd_delta - theta_t))
        weighted = torch.tensor(alpha_t, device=backend.device) * loss2
        loss = torch.mean(loss1.type(torch.float64) + weighted)
        loss.backward()
        opt.step()

        if len(buffer_losses) <= 300:
            buffer_losses.append(loss.detach())
            buffer_examples.append(to_numpy(adversarial))
        else:
            buffer_losses.pop(0)
            buffer_examples.pop(0)
            buffer_losses.append(loss.detach())
            buffer_examples.append(to_numpy(adversarial))

        matched = attack_succeeded(
            backend.decode(adversarial),
            as_transcript(target),
            targeted=True,
        )
        if step % 20 == 0 and matched:
            alpha_t *= 1.2
        if step % 50 == 0 and not matched:
            alpha_t *= 0.8

    index = int(torch.stack(buffer_losses).argmin().item())
    return buffer_examples[index]
