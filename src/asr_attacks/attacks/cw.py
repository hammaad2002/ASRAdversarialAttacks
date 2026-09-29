"""Carlini & Wagner (audio) and Qin et al. imperceptible / robust attacks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

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
from asr_attacks.text import display_text

# Qin et al. int16 hyperparameters converted to the [-1, 1] waveform scale.
_INT16 = 32768.0
_EPS_QIN = 2000.0 / _INT16  # ≈ 0.061
_LR1_QIN = 100.0 / _INT16  # ≈ 3.05e-3
_LR2_QIN = 1.0 / _INT16  # ≈ 3.05e-5
_LR_R1 = 50.0 / _INT16  # ≈ 1.5e-3
_LR_R2 = 5.0 / _INT16  # ≈ 1.5e-4
_LR_IR2 = 1.5 / _INT16  # ≈ 4.6e-5
_ROBUST_DELTA = 300.0 / _INT16  # ≈ 0.0092
_CW_EPS = 2000.0 / _INT16  # ≈ 0.061
_CW_LR = 10.0 / _INT16  # ≈ 3.05e-4


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
        loss = sum(
            _classifier_loss(
                backend,
                transform(adversarial),
                target_text,
                targeted=targeted,
                silence=silence,
                target_ids=target_ids,
            )
            for transform in transforms
        ) / float(m_rooms)
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
    ref_ids = backend.encode(target_text) if not targeted else target_ids
    ctc_ref: torch.Tensor | None = None
    if not targeted:
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
            loss_cls = sum(classifier(transform(adversarial)) for transform in transforms) / float(
                m_rooms
            )
            loss_theta = _psychoacoustic_loss(adversarial, original, theta_t, original_max_psd)
            (loss_cls + alpha_t * loss_theta).backward()
            opt.step()
            _project_inplace(adversarial, original, bound)

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
                alpha_t *= alpha_up
                l_theta_val = float(loss_theta.detach())
                if l_theta_val < best_theta:
                    best_theta = l_theta_val
                    best = adversarial.detach().clone()
            elif (step + 1) % 50 == 0:
                # IR2: paper has no decrease rule; package uses ×0.8 every 50.
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
