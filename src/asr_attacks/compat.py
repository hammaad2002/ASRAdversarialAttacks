"""Backward-compatible facade matching the original ``ASRAttacks`` method names."""

from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacker import ASRAttacker
from asr_attacks.backends.module import CTCModuleBackend


class ASRAttacks:
    """Compatibility wrapper around :class:`ASRAttacker`.

    The original notebook imported ``ASRAttacks`` and called ``FGSM_ATTACK``.
    New code should prefer :class:`asr_attacks.ASRAttacker`.
    """

    def __init__(self, model, device, labels: list[str] | tuple[str, ...]) -> None:
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
    ) -> np.ndarray:
        return self._attacker.fgsm(input__, target=target, epsilon=epsilon, targeted=targeted)

    def BIM_ATTACK(
        self,
        input__,
        target=None,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        num_iter: int = 10,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
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
        )

    def PGD_ATTACK(
        self,
        input__,
        target=None,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        num_iter: int = 40,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
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
        )

    def CW_ATTACK(
        self,
        input__,
        target=None,
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
        internal_call: bool = False,
    ) -> np.ndarray:
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
            optimizer=optimizer,
            nested=nested,
            early_stop=early_stop,
            search_eps=search_eps,
            targeted=targeted,
        )

    def IMPERCEPTIBLE_ATTACK(
        self,
        input__,
        target=None,
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
    ) -> np.ndarray:
        if target is None:
            raise ValueError("Please pass a target transcription for the imperceptible attack")
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
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            nested=nested,
            early_stop_cw=early_stop_cw,
            search_eps_cw=search_eps_cw,
            alpha=alpha,
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

        ``targeted`` is accepted for compatibility and ignored. Previous versions
        returned ``1 - WER`` for targeted evaluation; this wrapper always returns
        true WER.
        """
        del targeted
        return self._attacker.wer(ground_truth, audios)
