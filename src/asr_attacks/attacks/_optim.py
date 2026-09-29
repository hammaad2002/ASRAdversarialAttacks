"""Optimizer, projection and search plumbing shared by CW and Imperceptible."""

from __future__ import annotations

from collections.abc import Callable

import torch

from asr_attacks.attacks.common import ctc_loss, iteration_bar, maybe_early_stop
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded

# Qin et al. int16 hyperparameters are converted to the [-1, 1] waveform scale with this.
_INT16 = 32768.0


def _optimizer_for(name: str | None, parameters, learning_rate: float) -> torch.optim.Optimizer:
    key = "adam" if name is None else str(name).lower()
    if key == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if key == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer {name!r}; use 'adam' or 'sgd'")


def _project_inplace(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
) -> None:
    with torch.no_grad():
        delta = torch.clamp(adversarial - original, -epsilon, epsilon)
        adversarial.copy_(torch.clamp(original + delta, -1.0, 1.0))


def _clamp_audio_inplace(adversarial: torch.Tensor) -> None:
    with torch.no_grad():
        adversarial.clamp_(-1.0, 1.0)


def _silence_loss(logits: torch.Tensor, silence_token_ids: list[int]) -> torch.Tensor:
    """Per-frame hinge pushing mass onto silence tokens (CW Sec. III-F)."""
    if not silence_token_ids:
        raise ValueError("silence_ids() must return at least one token id")
    silence = torch.tensor(silence_token_ids, device=logits.device, dtype=torch.long)
    vocab = logits.shape[-1]
    mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)
    mask[silence] = False
    if not bool(mask.any()):
        raise ValueError("silence_ids() covers the entire vocabulary")
    max_silence = logits.index_select(-1, silence).max(dim=-1).values
    max_other = logits[..., mask].max(dim=-1).values
    return torch.relu(max_other - max_silence).sum(dim=-1).mean()


def _classifier_loss(
    backend: ASRBackend,
    adversarial: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    silence: bool,
    target_ids: torch.Tensor | None,
) -> torch.Tensor:
    logits = backend.logits(adversarial)
    if silence:
        return _silence_loss(logits, backend.silence_ids())
    assert target_ids is not None
    loss = ctc_loss(logits, target_ids, backend.blank_id)
    return loss if targeted else -loss


def _resolve_check_every(check_every: int | None, num_iter_decrease_eps: int) -> int:
    interval = num_iter_decrease_eps if check_every is None else check_every
    if interval <= 0:
        raise ValueError("check_every / num_iter_decrease_eps must be positive")
    return interval


def _bounded_search(
    backend: ASRBackend,
    original: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    epsilon: float,
    learning_rate: float,
    num_iter: int,
    decrease_factor_eps: float,
    check_every: int,
    optimizer: str | None,
    search_eps: bool,
    early_stop: bool,
    nested: bool,
    verbose: bool,
    desc: str | None,
    use_l2: bool,
    c: float,
    sign_grad: bool,
    silence: bool = False,
    room_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    success_fn: Callable[[torch.Tensor], bool] | None = None,
) -> tuple[torch.Tensor, float]:
    """L-inf box search with optional L2 / CTC objective and bound shrinking.

    Returns ``(best_or_final, final_epsilon)``. When ``search_eps`` finds a
    success, the returned waveform is the successful iterate from the smallest
    bound; otherwise it is the final iterate.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")

    adversarial = original.clone().requires_grad_(True)
    opt = _optimizer_for(optimizer, [adversarial], learning_rate)
    target_ids = None if silence else backend.encode(target_text)
    best: torch.Tensor | None = None
    current_eps = float(epsilon)

    def succeeded(wave: torch.Tensor) -> bool:
        if success_fn is not None:
            return success_fn(wave)
        return attack_succeeded(backend.decode(wave), target_text, targeted=targeted)

    success_message = (
        "Stopping early: targeted transcription reached."
        if targeted
        else "Stopping early: untargeted attack changed the transcription."
    )

    for step in iteration_bar(num_iter, nested=nested, desc=desc, verbose=verbose):
        opt.zero_grad()
        wave = room_transform(adversarial) if room_transform is not None else adversarial
        loss_cls = _classifier_loss(
            backend,
            wave,
            target_text,
            targeted=targeted,
            silence=silence,
            target_ids=target_ids,
        )
        if use_l2:
            loss = torch.sum((adversarial - original) ** 2) + c * loss_cls
        else:
            loss = loss_cls
        loss.backward()
        if sign_grad and adversarial.grad is not None:
            adversarial.grad.sign_()
        opt.step()
        _project_inplace(adversarial, original, current_eps)

        if maybe_early_stop(
            backend,
            adversarial,
            target_text,
            targeted=targeted,
            early_stop=early_stop,
            verbose=verbose,
            success_message=success_message,
        ):
            return adversarial.detach(), current_eps

        if search_eps and (step + 1) % check_every == 0 and succeeded(adversarial):
            best = adversarial.detach().clone()
            delta_max = float((adversarial.detach() - original).abs().max())
            current_eps = min(current_eps, delta_max) * decrease_factor_eps
            _project_inplace(adversarial, original, current_eps)

    if best is not None:
        return best, current_eps
    return adversarial.detach(), current_eps
