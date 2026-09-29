"""Wrapper around :class:`ASRAttacker` for a CTC module and its vocabulary."""

from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacker import ASRAttacker
from asr_attacks.attacks.cw import (
    _CW_EPS,
    _CW_LR,
    _EPS_QIN,
    _LR1_QIN,
    _LR2_QIN,
    _LR_IR2,
    _LR_R1,
    _LR_R2,
    _ROBUST_DELTA,
)
from asr_attacks.backends.module import CTCModuleBackend


class ASRAttacks:
    """Run the same attacks as :class:`~asr_attacks.ASRAttacker` on a CTC module.

    Args:
        model: Differentiable CTC model. Forward should return logits or
            ``(logits, extra)``.
        device: ``"cpu"``, ``"cuda"``, or a :class:`torch.device`.
        labels: Vocabulary in index order (for example ``bundle.get_labels()``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        labels: list[str] | tuple[str, ...],
    ) -> None:
        self.model = model
        self.device = device
        self.labels = labels
        self._attacker = ASRAttacker(CTCModuleBackend(model, labels=labels, device=device))

    def _encode_transcription(self, transcription):
        return self._attacker.backend.encode(transcription).detach().cpu()

    def FGSM_ATTACK(
        self,
        input__,
        target=None,
        epsilon: float = 0.2,
        targeted: bool = False,
        label=None,
        norm: float | str = "inf",
    ) -> np.ndarray:
        return self._attacker.fgsm(
            input__,
            target=target,
            epsilon=epsilon,
            targeted=targeted,
            label=label,
            norm=norm,
        )

    def BIM_ATTACK(
        self,
        input__,
        target=None,
        epsilon: float = 0.2,
        alpha: float | None = None,
        num_iter: int | None = None,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
        label=None,
        norm: float | str = "inf",
    ) -> np.ndarray:
        return self._attacker.bim(
            input__,
            target=target,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            nested=nested,
            targeted=targeted,
            early_stop=early_stop,
            label=label,
            norm=norm,
        )

    def PGD_ATTACK(
        self,
        input__,
        target=None,
        epsilon: float = 0.3,
        alpha: float | None = None,
        num_iter: int = 40,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
        label=None,
        norm: float | str = "inf",
        restarts: int = 1,
    ) -> np.ndarray:
        return self._attacker.pgd(
            input__,
            target=target,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            nested=nested,
            targeted=targeted,
            early_stop=early_stop,
            label=label,
            norm=norm,
            restarts=restarts,
        )

    def CW_ATTACK(
        self,
        input__,
        target=None,
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
        internal_call: bool = False,
        label=None,
        db_bound: float | None = None,
        loss: str = "ctc",
        kappa: float = 0.0,
    ) -> np.ndarray:
        """Audio Carlini & Wagner attack; see :meth:`ASRAttacker.cw`.

        ``num_iter_decrease_eps`` is used as ``check_every`` when
        ``check_every`` is ``None`` (legacy name kept for compatibility).
        """
        del internal_call
        return self._attacker.cw(
            input__,
            target=target,
            epsilon=epsilon,
            c=c,
            learning_rate=learning_rate,
            num_iter=num_iter,
            decrease_factor_eps=decrease_factor_eps,
            num_iter_decrease_eps=num_iter_decrease_eps,
            check_every=check_every,
            optimizer=optimizer,
            nested=nested,
            early_stop=early_stop,
            search_eps=search_eps,
            targeted=targeted,
            label=label,
            db_bound=db_bound,
            loss=loss,
            kappa=kappa,
        )

    def IMPERCEPTIBLE_ATTACK(
        self,
        input__,
        target=None,
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
        targeted: bool = True,
        label=None,
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
        """Qin et al. imperceptible / robust attack; see :meth:`ASRAttacker.imperceptible`.

        ``target`` may be ``None`` when ``targeted=False``. When
        ``targeted=True``, a missing ``target`` raises from
        ``resolve_target``. ``c`` is deprecated and ignored.
        """
        return self._attacker.imperceptible(
            input__,
            target,
            epsilon=epsilon,
            c=c,
            learning_rate1=learning_rate1,
            learning_rate2=learning_rate2,
            num_iter1=num_iter1,
            num_iter2=num_iter2,
            decrease_factor_eps=decrease_factor_eps,
            num_iter_decrease_eps=num_iter_decrease_eps,
            check_every=check_every,
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            nested=nested,
            early_stop_cw=early_stop_cw,
            search_eps_cw=search_eps_cw,
            alpha=alpha,
            sample_rate=sample_rate,
            targeted=targeted,
            label=label,
            mode=mode,
            rooms=rooms,
            robust_delta=robust_delta,
            num_iter_r1=num_iter_r1,
            num_iter_r2=num_iter_r2,
            learning_rate_r1=learning_rate_r1,
            learning_rate_r2=learning_rate_r2,
            num_iter_ir1=num_iter_ir1,
            num_iter_ir2=num_iter_ir2,
            learning_rate_ir1=learning_rate_ir1,
            learning_rate_ir2=learning_rate_ir2,
        )

    def INFER(self, input_: torch.Tensor) -> str:
        return self._attacker.decode(input_)

    def wer_compute(
        self,
        ground_truth: list[str],
        audios: list[np.ndarray],
        targeted: bool = False,
    ) -> tuple[float, list[tuple[int, int, int]]]:
        """Return mean Levenshtein WER and per-utterance (S, I, D) counts.

        ``targeted`` is unused. This method always returns WER.
        """
        del targeted
        return self._attacker.wer(ground_truth, audios)
