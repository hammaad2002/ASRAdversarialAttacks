"""Carlini & Wagner (audio) attack, Sec. III-B/C/F (arXiv:1801.01944).

The Qin et al. imperceptible / robust attacks live in :mod:`asr_attacks.attacks.imperceptible`
and are re-exported here for backward compatibility.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from asr_attacks.attacks._optim import (
    _INT16,
    _bounded_search,
    _optimizer_for,
    _project_inplace,
    _resolve_check_every,
)
from asr_attacks.attacks.common import iteration_bar, resolve_target
from asr_attacks.attacks.imperceptible import imperceptible
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.tensors import prepare_audio, to_numpy
from asr_attacks.text import display_text

__all__ = ["cw", "imperceptible"]

_CW_EPS = 2000.0 / _INT16  # ≈ 0.061
_CW_LR = 10.0 / _INT16  # ≈ 3.05e-4


def _margin_frame_loss(
    logits: torch.Tensor,
    alignment: torch.Tensor,
    frame_weights: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Per-frame CW hinge on logits for a fixed greedy alignment (Sec. III-C)."""
    # logits: (1, T, V); alignment / weights: (T,)
    time = logits.shape[1]
    if alignment.numel() != time:
        raise ValueError(f"Alignment length {alignment.numel()} does not match logit frames {time}")
    rows = torch.arange(time, device=logits.device)
    correct = logits[0, rows, alignment]
    wrong = logits[0].clone()
    wrong[rows, alignment] = float("-inf")
    hinge = torch.relu(wrong.max(dim=-1).values - correct + kappa)
    return (frame_weights * hinge).sum()


def _cw_margin_stage(
    backend: ASRBackend,
    original: torch.Tensor,
    stage_one: torch.Tensor,
    *,
    epsilon: float,
    c: float,
    learning_rate: float,
    num_iter: int,
    decrease_factor_eps: float,
    check_every: int,
    optimizer: str | None,
    search_eps: bool,
    early_stop: bool,
    nested: bool,
    verbose: bool,
    kappa: float,
) -> torch.Tensor:
    with torch.no_grad():
        alignment = torch.argmax(backend.logits(stage_one)[0], dim=-1).detach()

    adversarial = stage_one.clone().detach().requires_grad_(True)
    opt = _optimizer_for(optimizer, [adversarial], learning_rate)
    frame_c = torch.full((alignment.numel(),), float(c), device=original.device)
    best: torch.Tensor | None = None
    current_eps = float(epsilon)

    for step in iteration_bar(num_iter, nested=nested, desc="CW margin", verbose=verbose):
        opt.zero_grad()
        logits = backend.logits(adversarial)
        loss = torch.sum((adversarial - original) ** 2) + _margin_frame_loss(
            logits, alignment, frame_c, kappa
        )
        loss.backward()
        opt.step()
        _project_inplace(adversarial, original, current_eps)

        with torch.no_grad():
            pred = torch.argmax(backend.logits(adversarial)[0], dim=-1)
            matched_alignment = bool(torch.equal(pred, alignment))

        if early_stop and matched_alignment:
            if verbose:
                print("Stopping early: margin alignment matched.")
            return adversarial.detach()

        if search_eps and (step + 1) % check_every == 0:
            if matched_alignment:
                best = adversarial.detach().clone()
                delta_max = float((adversarial.detach() - original).abs().max())
                current_eps = min(current_eps, delta_max) * decrease_factor_eps
                _project_inplace(adversarial, original, current_eps)
            # Package choice (paper gives no update rule): double c_i on wrong frames.
            wrong = pred != alignment
            frame_c = torch.where(wrong, frame_c * 2.0, frame_c)

    return best if best is not None else adversarial.detach()


def cw(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    *,
    epsilon: float = _CW_EPS,
    c: float = 1.0,
    learning_rate: float = _CW_LR,
    num_iter: int = 5000,
    decrease_factor_eps: float = 0.8,
    num_iter_decrease_eps: int = 10,
    check_every: int | None = None,
    optimizer: str | None = "adam",
    nested: bool = True,
    early_stop: bool = False,
    search_eps: bool = True,
    targeted: bool = False,
    target: str | list[str] | None = None,
    label: str | list[str] | None = None,
    db_bound: float | None = None,
    loss: str = "ctc",
    kappa: float = 0.0,
    verbose: bool = True,
    desc: str | None = None,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """Audio Carlini & Wagner attack ([arXiv:1801.01944](https://arxiv.org/abs/1801.01944)).

    Sec. III-B minimizes ``||delta||_2^2 + c * CTC(x+delta, t)`` inside an L-inf
    box (``epsilon``, or ``db_bound``), with Adam and paper-style bound shrinking.
    Untargeted mode flips the CTC sign and is a package extension.

    Sec. III-C (``loss="margin"``, targeted only) refines a CTC solution with a
    per-frame hinge on logits under a fixed greedy alignment.

    Sec. III-F allows ``target=""``: a silence-token hinge replaces CTC, and
    success is an empty displayed transcript.
    """
    if early_stop and search_eps:
        raise ValueError("early_stop and search_eps cannot both be True")
    loss_key = str(loss).lower()
    if loss_key not in {"ctc", "margin"}:
        raise ValueError("loss must be 'ctc' or 'margin'")
    if loss_key == "margin" and not targeted:
        raise ValueError("loss='margin' is targeted only")

    original = prepare_audio(audio, backend.device).detach()
    if db_bound is not None:
        peak = float(original.abs().max())
        if peak <= 0.0:
            raise ValueError("db_bound requires a non-zero peak amplitude")
        epsilon = peak * (10.0 ** (float(db_bound) / 20.0))
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")

    target_text = resolve_target(backend, original, target, targeted, label)
    silence = targeted and display_text(target_text) == ""
    interval = _resolve_check_every(check_every, num_iter_decrease_eps)

    stage_one, eps_after = _bounded_search(
        backend,
        original,
        target_text,
        targeted=targeted,
        epsilon=epsilon,
        learning_rate=learning_rate,
        num_iter=num_iter,
        decrease_factor_eps=decrease_factor_eps,
        check_every=interval,
        optimizer=optimizer,
        search_eps=search_eps,
        early_stop=early_stop,
        nested=nested,
        verbose=verbose,
        desc=desc,
        use_l2=True,
        c=c,
        sign_grad=False,
        silence=silence,
    )

    if loss_key == "ctc":
        if return_tensor:
            return stage_one
        return to_numpy(stage_one)

    if not attack_succeeded(backend.decode(stage_one), target_text, targeted=True):
        warnings.warn(
            "CW margin stage skipped: CTC stage did not reach the target transcript.",
            UserWarning,
            stacklevel=2,
        )
        if return_tensor:
            return stage_one
        return to_numpy(stage_one)

    refined = _cw_margin_stage(
        backend,
        original,
        stage_one,
        epsilon=eps_after if search_eps else epsilon,
        c=c,
        learning_rate=learning_rate,
        num_iter=num_iter,
        decrease_factor_eps=decrease_factor_eps,
        check_every=interval,
        optimizer=optimizer,
        search_eps=search_eps,
        early_stop=early_stop,
        nested=nested,
        verbose=verbose,
        kappa=kappa,
    )
    if return_tensor:
        return refined
    return to_numpy(refined)
