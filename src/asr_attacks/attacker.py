from __future__ import annotations

import numpy as np
import torch

from asr_attacks.attacks.cw import cw, imperceptible
from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.iterative import bim, pgd
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import alignment_counts, word_error_rate
from asr_attacks.tensors import prepare_audio


class ASRAttacker:
    """Run white-box CTC attacks against an :class:`ASRBackend`."""

    def __init__(self, backend: ASRBackend, verbose: bool = True) -> None:
        self.backend = backend
        self.verbose = verbose

    def decode(self, audio: torch.Tensor | np.ndarray) -> str:
        return self.backend.decode(prepare_audio(audio, self.backend.device))

    def fgsm(self, audio, target=None, epsilon: float = 0.2, targeted: bool = False) -> np.ndarray:
        return fgsm(self.backend, audio, epsilon=epsilon, targeted=targeted, target=target)

    def bim(
        self,
        audio,
        target=None,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        num_iter: int = 10,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
    ) -> np.ndarray:
        return bim(
            self.backend,
            audio,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            targeted=targeted,
            target=target,
            nested=nested,
            early_stop=early_stop,
            verbose=self.verbose,
        )

    def pgd(
        self,
        audio,
        target=None,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        num_iter: int = 40,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
        random_start: bool = True,
    ) -> np.ndarray:
        return pgd(
            self.backend,
            audio,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            targeted=targeted,
            target=target,
            nested=nested,
            early_stop=early_stop,
            random_start=random_start,
            verbose=self.verbose,
        )

    def cw(
        self,
        audio,
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
    ) -> np.ndarray:
        result = cw(
            self.backend,
            audio,
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
            target=target,
            verbose=self.verbose,
        )
        assert isinstance(result, np.ndarray)
        return result

    def imperceptible(
        self,
        audio,
        target,
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
    ) -> np.ndarray:
        return imperceptible(
            self.backend,
            audio,
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
            sample_rate=sample_rate,
            verbose=self.verbose,
        )

    def wer(
        self,
        references: list[str],
        audios: list[np.ndarray],
    ) -> tuple[float, list[tuple[int, int, int]]]:
        scores: list[float] = []
        counts: list[tuple[int, int, int]] = []
        for reference, sample in zip(references, audios, strict=True):
            hypothesis = self.decode(sample)
            scores.append(word_error_rate(reference, hypothesis))
            counts.append(alignment_counts(reference, hypothesis))
        return float(sum(scores) / len(scores)), counts
