"""Imperceptible (Qin et al. 2019, arXiv:1903.10346) and robust over-the-air attacks."""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from asr_attacks.attacks._optim import (
    _INT16,
    _bounded_search,
    _clamp_audio_inplace,
    _classifier_loss,
    _optimizer_for,
    _project_inplace,
    _resolve_check_every,
    _silence_loss,
)
from asr_attacks.attacks.common import ctc_loss, iteration_bar, resolve_target
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.psychoacoustic import compute_masking_threshold, psd_transform
from asr_attacks.tensors import prepare_audio, to_numpy
from asr_attacks.text import display_text

# Qin et al. int16 hyperparameters converted to the [-1, 1] waveform scale.
_EPS_QIN = 2000.0 / _INT16  # ≈ 0.061
_LR1_QIN = 100.0 / _INT16  # ≈ 3.05e-3
_LR2_QIN = 1.0 / _INT16  # ≈ 3.05e-5
_LR_R1 = 50.0 / _INT16  # ≈ 1.5e-3
_LR_R2 = 5.0 / _INT16  # ≈ 1.5e-4
_LR_IR2 = 1.5 / _INT16  # ≈ 4.6e-5
_ROBUST_DELTA = 300.0 / _INT16  # ≈ 0.0092

# Imperceptible+robust alpha schedule: weaker penalty every _IR_DECREASE_EVERY steps without a
# success (the offline stage 2 uses the same 50-step rule).
_IR_DECREASE_EVERY = 50


def _psychoacoustic_loss(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    theta_t: torch.Tensor,
    original_max_psd: np.ndarray | float,
) -> torch.Tensor:
    perturbation = adversarial - original
    psd_delta = psd_transform(perturbation, original_max_psd, adversarial.device)
    return torch.mean(torch.nn.functional.relu(psd_delta - theta_t))


def _imperceptible_stage2(
    backend: ASRBackend,
    original: torch.Tensor,
    stage_one: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    learning_rate2: float,
    num_iter2: int,
    optimizer2: str | None,
    alpha: float,
    sample_rate: int,
    nested: bool,
    verbose: bool,
    ctc_ref_stage1: torch.Tensor | None,
    silence: bool = False,
) -> np.ndarray:
    adversarial = stage_one.clone().detach().requires_grad_(True)
    opt_name = optimizer2 if optimizer2 is not None else "adam"
    opt = _optimizer_for(opt_name, [adversarial], learning_rate2)
    target_ids = None if silence or not targeted else backend.encode(target_text)
    ref_ids = None if silence else backend.encode(target_text)
    waveform = to_numpy(original).squeeze()
    theta, original_max_psd = compute_masking_threshold(waveform, sample_rate=sample_rate)
    theta_t = torch.tensor(theta.transpose(1, 0), device=backend.device)
    alpha_t = float(alpha)
    best_example: np.ndarray | None = None
    best_l_theta = math.inf
    lr_decayed = False

    for step in iteration_bar(
        num_iter2, nested=nested, desc="*****Attack Stage 2*****", verbose=verbose
    ):
        if not lr_decayed and step == 3000:
            for group in opt.param_groups:
                group["lr"] *= 0.1
            lr_decayed = True

        opt.zero_grad()
        logits = backend.logits(adversarial)
        if silence:
            loss_classifier = _silence_loss(logits, backend.silence_ids())
        elif targeted:
            assert target_ids is not None
            loss_classifier = ctc_loss(logits, target_ids, backend.blank_id)
        else:
            assert ctc_ref_stage1 is not None and ref_ids is not None
            loss_classifier = torch.relu(
                ctc_ref_stage1 - ctc_loss(logits, ref_ids, backend.blank_id)
            )
        loss_theta = _psychoacoustic_loss(adversarial, original, theta_t, original_max_psd)
        loss = loss_classifier + alpha_t * loss_theta
        loss.backward()
        opt.step()
        _clamp_audio_inplace(adversarial)

        check_success = (step + 1) % 20 == 0
        check_fail = (step + 1) % 50 == 0
        if check_success or check_fail:
            matched = attack_succeeded(
                backend.decode(adversarial),
                target_text,
                targeted=targeted,
            )
            if check_success and matched:
                alpha_t *= 1.2
            if check_fail and not matched:
                alpha_t *= 0.8
            if matched:
                l_theta_val = float(loss_theta.detach())
                if l_theta_val < best_l_theta:
                    best_l_theta = l_theta_val
                    best_example = to_numpy(adversarial)

    if best_example is not None:
        return best_example
    return to_numpy(stage_one)


def _robust_success(
    backend: ASRBackend,
    adversarial: torch.Tensor,
    target_text: str,
    rooms,
    *,
    targeted: bool,
    m: int,
    require_all: bool,
) -> bool:
    transforms = rooms.sample_set(m)
    hits = 0
    for transform in transforms:
        hyp = backend.decode(transform(adversarial))
        if attack_succeeded(hyp, target_text, targeted=targeted):
            hits += 1
            if not require_all:
                return True
        elif require_all:
            return False
    return hits == m if require_all else hits > 0


def _run_robust(
    backend: ASRBackend,
    original: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    rooms,
    epsilon: float,
    robust_delta: float,
    decrease_factor_eps: float,
    check_every: int,
    nested: bool,
    verbose: bool,
    num_iter_r1: int,
    num_iter_r2: int,
    learning_rate_r1: float,
    learning_rate_r2: float,
    optimizer: str | None,
    m_rooms: int = 10,
) -> tuple[torch.Tensor, float]:
    def r1_success(wave: torch.Tensor) -> bool:
        return _robust_success(
            backend, wave, target_text, rooms, targeted=targeted, m=1, require_all=False
        )

    # R1: bound shrinks; transform resampled every step via a fresh sample in the loop.
    adversarial = original.clone().requires_grad_(True)
    opt_name = optimizer if optimizer is not None else "sgd"
    opt = _optimizer_for(opt_name, [adversarial], learning_rate_r1)
    best: torch.Tensor | None = None
    current_eps = float(epsilon)
    silence = targeted and display_text(target_text) == ""
    target_ids = None if silence else backend.encode(target_text)

    for step in iteration_bar(
        num_iter_r1, nested=nested, desc="*****Robust R1*****", verbose=verbose
    ):
        opt.zero_grad()
        transformed = rooms.sample()(adversarial)
        loss = _classifier_loss(
            backend,
            transformed,
            target_text,
            targeted=targeted,
            silence=silence,
            target_ids=target_ids,
        )
        loss.backward()
        if adversarial.grad is not None:
            adversarial.grad.sign_()
        opt.step()
        _project_inplace(adversarial, original, current_eps)

        if (step + 1) % check_every == 0 and r1_success(adversarial):
            best = adversarial.detach().clone()
            delta_max = float((adversarial.detach() - original).abs().max())
            current_eps = min(current_eps, delta_max) * decrease_factor_eps
            _project_inplace(adversarial, original, current_eps)

    stage_r1 = best if best is not None else adversarial.detach()
    eps_r = float((stage_r1 - original).abs().max())
    bound = eps_r + float(robust_delta)

    # R2: fixed bound; success = all M rooms.
    adversarial = stage_r1.clone().detach().requires_grad_(True)
    opt = _optimizer_for(opt_name, [adversarial], learning_rate_r2)
    best_r2: torch.Tensor | None = None

    for step in iteration_bar(
        num_iter_r2, nested=nested, desc="*****Robust R2*****", verbose=verbose
    ):
        opt.zero_grad()
        transforms = rooms.sample_set(m_rooms)
        loss = torch.stack(
            [
                _classifier_loss(
                    backend,
                    transform(adversarial),
                    target_text,
                    targeted=targeted,
                    silence=silence,
                    target_ids=target_ids,
                )
                for transform in transforms
            ]
        ).mean()
        loss.backward()
        if adversarial.grad is not None:
            adversarial.grad.sign_()
        opt.step()
        _project_inplace(adversarial, original, bound)

        if (step + 1) % check_every == 0 and _robust_success(
            backend,
            adversarial,
            target_text,
            rooms,
            targeted=targeted,
            m=m_rooms,
            require_all=True,
        ):
            best_r2 = adversarial.detach().clone()

    return (best_r2 if best_r2 is not None else adversarial.detach()), bound


def _run_imperceptible_robust(
    backend: ASRBackend,
    original: torch.Tensor,
    robust_start: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    rooms,
    bound: float,
    sample_rate: int,
    nested: bool,
    verbose: bool,
    check_every: int,
    num_iter_ir1: int,
    num_iter_ir2: int,
    learning_rate_ir1: float,
    learning_rate_ir2: float,
    m_rooms: int = 10,
) -> np.ndarray:
    waveform = to_numpy(original).squeeze()
    theta, original_max_psd = compute_masking_threshold(waveform, sample_rate=sample_rate)
    theta_t = torch.tensor(theta.transpose(1, 0), device=backend.device)
    silence = targeted and display_text(target_text) == ""
    target_ids = None if silence else backend.encode(target_text)
    ref_ids: torch.Tensor | None = None
    ctc_ref: torch.Tensor | None = None
    if not targeted:
        ref_ids = backend.encode(target_text)
        with torch.no_grad():
            ctc_ref = ctc_loss(backend.logits(robust_start), ref_ids, backend.blank_id).detach()

    def classifier(wave: torch.Tensor) -> torch.Tensor:
        if silence:
            return _silence_loss(backend.logits(wave), backend.silence_ids())
        logits = backend.logits(wave)
        if targeted:
            assert target_ids is not None
            return ctc_loss(logits, target_ids, backend.blank_id)
        assert ctc_ref is not None and ref_ids is not None
        return torch.relu(ctc_ref - ctc_loss(logits, ref_ids, backend.blank_id))

    def ir_loop(
        start: torch.Tensor,
        *,
        alpha0: float,
        lr: float,
        num_iter: int,
        success_rooms: int,
        alpha_up: float,
        alpha_down: float,
        desc: str,
    ) -> torch.Tensor:
        adversarial = start.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([adversarial], lr=lr)
        alpha_t = float(alpha0)
        best = adversarial.detach().clone()
        best_theta = math.inf

        for step in iteration_bar(num_iter, nested=nested, desc=desc, verbose=verbose):
            opt.zero_grad()
            transforms = rooms.sample_set(m_rooms)
            loss_cls = torch.stack(
                [classifier(transform(adversarial)) for transform in transforms]
            ).mean()
            loss_theta = _psychoacoustic_loss(adversarial, original, theta_t, original_max_psd)
            (loss_cls + alpha_t * loss_theta).backward()
            opt.step()
            _project_inplace(adversarial, original, bound)

            # Decoding every room is the expensive part, so only look at the schedule's
            # check steps (success -> stronger penalty) and 50-step boundaries (no success ->
            # weaker penalty), like the offline stage 2.
            check_success = (step + 1) % check_every == 0
            check_fail = (step + 1) % _IR_DECREASE_EVERY == 0
            if not (check_success or check_fail):
                continue
            hits = sum(
                1
                for transform in transforms
                if attack_succeeded(
                    backend.decode(transform(adversarial)),
                    target_text,
                    targeted=targeted,
                )
            )
            if hits >= success_rooms:
                if check_success:
                    alpha_t *= alpha_up
                l_theta_val = float(loss_theta.detach())
                if l_theta_val < best_theta:
                    best_theta = l_theta_val
                    best = adversarial.detach().clone()
            elif check_fail:
                # IR2: paper has no decrease rule; package uses x0.8 every 50.
                alpha_t *= alpha_down

        return best

    after_ir1 = ir_loop(
        robust_start,
        alpha0=0.01,
        lr=learning_rate_ir1,
        num_iter=num_iter_ir1,
        success_rooms=4,
        alpha_up=2.0,
        alpha_down=0.5,
        desc="*****IR1*****",
    )
    after_ir2 = ir_loop(
        after_ir1,
        alpha0=5e-5,
        lr=learning_rate_ir2,
        num_iter=num_iter_ir2,
        success_rooms=8,
        alpha_up=1.2,
        alpha_down=0.8,
        desc="*****IR2*****",
    )
    return to_numpy(after_ir2)


def imperceptible(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    target: str | list[str] | None = None,
    *,
    epsilon: float = _EPS_QIN,
    c: float | None = None,
    learning_rate1: float = _LR1_QIN,
    learning_rate2: float = _LR2_QIN,
    num_iter1: int = 1000,
    num_iter2: int = 4000,
    decrease_factor_eps: float = 0.8,
    num_iter_decrease_eps: int = 10,
    check_every: int | None = None,
    optimizer1: str | None = "sgd",
    optimizer2: str | None = "adam",
    nested: bool = True,
    early_stop_cw: bool = False,
    search_eps_cw: bool = True,
    alpha: float = 0.05,
    sample_rate: int = 16000,
    verbose: bool = True,
    targeted: bool = True,
    label: str | list[str] | None = None,
    mode: str = "imperceptible",
    rooms=None,
    robust_delta: float = _ROBUST_DELTA,
    num_iter_r1: int = 2000,
    num_iter_r2: int = 4000,
    learning_rate_r1: float = _LR_R1,
    learning_rate_r2: float = _LR_R2,
    num_iter_ir1: int = 4000,
    num_iter_ir2: int = 6000,
    learning_rate_ir1: float = _LR2_QIN,
    learning_rate_ir2: float = _LR_IR2,
) -> np.ndarray:
    """Qin et al. imperceptible / robust ASR attack ([arXiv:1903.10346](https://arxiv.org/abs/1903.10346)).

    ``mode="imperceptible"`` (default) follows Sec. 4 offline: stage 1 is a
    CTC-only L-inf bound search with signed gradient steps (Algorithm 1);
    stage 2 minimizes ``CTC + alpha * l_theta`` with Adam, waveform clamped to
    ``[-1, 1]``. The paper's Lingvo model uses cross-entropy; CTC backends use
    CTC for ``l_net``.

    ``mode="robust"`` / ``"imperceptible_robust"`` need ``rooms=`` (see
    :mod:`asr_attacks.rooms`). IR2 uses a package choice of ``alpha *= 0.8``
    every 50 failing iterations (the paper gives no decrease rule).

    Untargeted mode (``targeted=False``) is a package extension: stage 1
    maximizes CTC against the reference; stage 2 uses a hinge on the stage-1
    CTC value.

    ``c`` is deprecated and ignored (with a warning) when not ``None``.
    """
    mode_key = str(mode).lower()
    if mode_key not in {"imperceptible", "robust", "imperceptible_robust"}:
        raise ValueError("mode must be 'imperceptible', 'robust', or 'imperceptible_robust'")
    if mode_key in {"robust", "imperceptible_robust"} and rooms is None:
        raise ValueError(f"mode={mode_key!r} requires rooms=")
    if c is not None:
        warnings.warn(
            "imperceptible(c=...) is deprecated and ignored; stage 2 no longer uses c.",
            DeprecationWarning,
            stacklevel=2,
        )

    original = prepare_audio(audio, backend.device).detach()
    target_text = resolve_target(backend, original, target, targeted, label)
    interval = _resolve_check_every(check_every, num_iter_decrease_eps)
    silence = targeted and display_text(target_text) == ""
    opt1 = "sgd" if optimizer1 is None else optimizer1
    opt2 = "adam" if optimizer2 is None else optimizer2

    if mode_key in {"robust", "imperceptible_robust"}:
        robust_wave, bound = _run_robust(
            backend,
            original,
            target_text,
            targeted=targeted,
            rooms=rooms,
            epsilon=epsilon,
            robust_delta=robust_delta,
            decrease_factor_eps=decrease_factor_eps,
            check_every=interval,
            nested=nested,
            verbose=verbose,
            num_iter_r1=num_iter_r1,
            num_iter_r2=num_iter_r2,
            learning_rate_r1=learning_rate_r1,
            learning_rate_r2=learning_rate_r2,
            optimizer=opt1,
        )
        if mode_key == "robust":
            return to_numpy(robust_wave)
        return _run_imperceptible_robust(
            backend,
            original,
            robust_wave,
            target_text,
            targeted=targeted,
            rooms=rooms,
            bound=bound,
            sample_rate=sample_rate,
            nested=nested,
            verbose=verbose,
            check_every=interval,
            num_iter_ir1=num_iter_ir1,
            num_iter_ir2=num_iter_ir2,
            learning_rate_ir1=learning_rate_ir1,
            learning_rate_ir2=learning_rate_ir2,
        )

    # Offline imperceptible (Sec. 4).
    if early_stop_cw and search_eps_cw:
        raise ValueError("early_stop_cw and search_eps_cw cannot both be True")

    stage_one, _ = _bounded_search(
        backend,
        original,
        target_text,
        targeted=targeted,
        epsilon=epsilon,
        learning_rate=learning_rate1,
        num_iter=num_iter1,
        decrease_factor_eps=decrease_factor_eps,
        check_every=interval,
        optimizer=opt1,
        search_eps=search_eps_cw,
        early_stop=early_stop_cw,
        nested=nested,
        verbose=verbose,
        desc="*****Attack Stage 1*****",
        use_l2=False,
        c=1.0,
        sign_grad=str(opt1).lower() == "sgd",
        silence=silence,
    )

    ctc_ref_stage1 = None
    if not targeted and not silence:
        with torch.no_grad():
            ctc_ref_stage1 = ctc_loss(
                backend.logits(stage_one),
                backend.encode(target_text),
                backend.blank_id,
            ).detach()

    return _imperceptible_stage2(
        backend,
        original,
        stage_one,
        target_text,
        targeted=targeted,
        learning_rate2=learning_rate2,
        num_iter2=num_iter2,
        optimizer2=opt2,
        alpha=alpha,
        sample_rate=sample_rate,
        nested=nested,
        verbose=verbose,
        ctc_ref_stage1=ctc_ref_stage1,
        silence=silence,
    )
