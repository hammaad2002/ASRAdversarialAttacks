from __future__ import annotations

import numpy as np
import torch

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
    cw,
    imperceptible,
)
from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.iterative import bim, pgd
from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import alignment_counts, word_error_rate
from asr_attacks.tensors import prepare_audio


class ASRAttacker:
    """Run white-box CTC attacks against an :class:`~asr_attacks.backends.ASRBackend`.

    Args:
        backend: Model wrapper that can encode targets, decode audio, and
            return differentiable CTC logits.
        verbose: Print early-stop messages from iterative attacks.
    """

    def __init__(self, backend: ASRBackend, verbose: bool = True) -> None:
        self.backend = backend
        self.verbose = verbose

    def decode(self, audio: torch.Tensor | np.ndarray) -> str:
        """Greedy CTC decode of a waveform.

        Args:
            audio: One utterance, shape ``(samples,)`` or ``(1, samples)``.

        Returns:
            Transcript string. wav2vec2-style word boundaries appear as ``|``.
        """
        return self.backend.decode(prepare_audio(audio, self.backend.device))

    def fgsm(
        self,
        audio: torch.Tensor | np.ndarray,
        target: str | list[str] | None = None,
        epsilon: float = 0.2,
        targeted: bool = False,
        label: str | list[str] | None = None,
        norm: float | str = "inf",
    ) -> np.ndarray:
        """One-step Fast Gradient Sign / Fast Gradient Method.

        ``x_adv = clip_[-1, 1](x + epsilon * s * normalize_p(grad_x CTC(f(x), y)))``
        with ``s = +1`` untargeted and ``s = -1`` targeted. ``normalize_p`` is
        ``sign(g)`` (L-inf, FGSM of [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)),
        ``g / ||g||_2`` or ``g / ||g||_1`` (L2 / L1 Fast Gradient Method of
        [Kurakin et al., 2017](https://arxiv.org/abs/1611.01236); L1 follows ART).
        Targeted FGSM is also from Kurakin et al., 2017.

        Args:
            audio: One utterance, shape ``(samples,)`` or ``(1, samples)``.
            target: Target transcript when ``targeted=True``. A string, a
                character list, or a one-element sentence list.
            epsilon: Step size, measured in the chosen ``norm``.
            targeted: If true, minimize CTC loss toward ``target``. Otherwise
                maximize loss away from ``label`` (or the clean greedy decode).
            label: Ground-truth transcript for untargeted attacks. Defaults to
                the model's clean prediction. Raises ``ValueError`` with
                ``targeted=True``.
            norm: ``"inf"`` (or ``np.inf``), ``2`` or ``1``.

        Returns:
            CPU NumPy waveform, clipped to ``[-1, 1]``.
        """
        return fgsm(
            self.backend,
            audio,
            epsilon=epsilon,
            targeted=targeted,
            target=target,
            label=label,
            norm=norm,
        )

    def bim(
        self,
        audio: torch.Tensor | np.ndarray,
        target: str | list[str] | None = None,
        epsilon: float = 0.2,
        alpha: float | None = None,
        num_iter: int | None = None,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
        label: str | list[str] | None = None,
        norm: float | str = "inf",
    ) -> np.ndarray:
        """Basic Iterative Method ([Kurakin et al., 2017](https://arxiv.org/abs/1607.02533)).

        Iterative signed/normalized gradient steps projected onto the L-``norm``
        ball of radius ``epsilon``, then clipped to ``[-1, 1]``.

        Args:
            audio: Clean waveform.
            target: Target transcript when ``targeted=True``.
            epsilon: Maximum perturbation in the chosen ``norm``.
            alpha: Per-step size. Defaults to ``epsilon / 10``.
            num_iter: Steps. Defaults to the paper rule
                ``ceil(min(eps/alpha + 4, 1.25 * eps/alpha))``.
            nested: If true, hide the progress bar when this call sits inside
                another tqdm loop.
            targeted: Targeted vs untargeted objective.
            early_stop: Stop when the greedy decode matches (targeted) or
                differs from (untargeted) the reference transcript.
            label: Ground-truth transcript for untargeted attacks, used by the
                loss and the early-stop check. Defaults to the clean decode.
            norm: ``"inf"`` (or ``np.inf``), ``2``, or ``1``.

        Returns:
            CPU NumPy waveform.
        """
        return bim(
            self.backend,
            audio,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            targeted=targeted,
            target=target,
            label=label,
            nested=nested,
            early_stop=early_stop,
            verbose=self.verbose,
            norm=norm,
        )

    def pgd(
        self,
        audio: torch.Tensor | np.ndarray,
        target: str | list[str] | None = None,
        epsilon: float = 0.3,
        alpha: float | None = None,
        num_iter: int = 40,
        nested: bool = True,
        targeted: bool = False,
        early_stop: bool = False,
        random_start: bool = True,
        label: str | list[str] | None = None,
        norm: float | str = "inf",
        restarts: int = 1,
    ) -> np.ndarray:
        """Projected Gradient Descent ([Madry et al., 2018](https://arxiv.org/abs/1706.06083)).

        Same inner loop as BIM with an optional random start inside the
        L-``norm`` ball. Pass ``random_start=False`` for BIM-style init.

        Args:
            audio: Clean waveform.
            target: Target transcript when ``targeted=True``.
            epsilon: Maximum perturbation in the chosen ``norm``.
            alpha: Per-step size. Defaults to ``2.5 * epsilon / num_iter``.
            num_iter: Number of gradient steps.
            nested: Hide progress bar when nested in another loop.
            targeted: Targeted vs untargeted objective.
            early_stop: Stop on transcription success.
            random_start: Sample the initial point inside the L-``norm`` ball.
            label: Ground-truth transcript for untargeted attacks, used by the
                loss and the early-stop check. Defaults to the clean decode.
            norm: ``"inf"`` (or ``np.inf``), ``2``, or ``1``.
            restarts: Independent random starts; keep the strongest. Requires
                ``random_start=True`` when ``restarts > 1``.

        Returns:
            CPU NumPy waveform.
        """
        return pgd(
            self.backend,
            audio,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            targeted=targeted,
            target=target,
            label=label,
            nested=nested,
            early_stop=early_stop,
            random_start=random_start,
            verbose=self.verbose,
            norm=norm,
            restarts=restarts,
        )

    def cw(
        self,
        audio: torch.Tensor | np.ndarray,
        target: str | list[str] | None = None,
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
        label: str | list[str] | None = None,
        db_bound: float | None = None,
        loss: str = "ctc",
        kappa: float = 0.0,
    ) -> np.ndarray:
        """Audio Carlini & Wagner attack ([arXiv:1801.01944](https://arxiv.org/abs/1801.01944)).

        Sec. III-B minimizes ``||delta||_2^2 + c * CTC(x+delta, t)`` inside an
        L-inf box (``epsilon``, or ``db_bound``), with Adam and paper-style
        bound shrinking. Untargeted mode flips the CTC sign (package extension).

        Sec. III-C (``loss="margin"``, targeted only) refines a CTC solution
        with a per-frame hinge on logits. Sec. III-F allows ``target=""``
        (silence-token hinge; success is an empty transcript).

        Args:
            audio: Clean waveform.
            target: Target transcript when ``targeted=True`` (``""`` for silence).
            epsilon: Absolute L-inf box (default ≈ 2000/32768). Ignored when
                ``db_bound`` is set.
            c: Weight on the CTC (or silence / margin) term. Default ``1.0``.
            learning_rate: Optimizer step size (default ≈ 10/32768).
            num_iter: Optimization steps (default 5000).
            decrease_factor_eps: Multiplier applied to ``epsilon`` on success
                during ``search_eps`` (default 0.8).
            num_iter_decrease_eps: Used as ``check_every`` when ``check_every``
                is ``None``.
            check_every: Decode / bound-shrink interval. Defaults to
                ``num_iter_decrease_eps``.
            optimizer: ``"adam"`` (default) or ``"sgd"``.
            nested: Hide progress bar when nested in another loop.
            early_stop: Stop on transcription success. Cannot be combined with
                ``search_eps``.
            search_eps: Shrink ``epsilon`` after successes and return the best
                successful iterate (default ``True``).
            targeted: Targeted vs untargeted objective.
            label: Ground-truth transcript for untargeted attacks. Defaults to
                the clean decode.
            db_bound: If set, ``epsilon = max|x| * 10**(db_bound/20)``.
            loss: ``"ctc"`` (Sec. III-B) or ``"margin"`` (Sec. III-C, targeted).
            kappa: Margin in the Sec. III-C hinge.

        Returns:
            CPU NumPy waveform.
        """
        result = cw(
            self.backend,
            audio,
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
            target=target,
            label=label,
            db_bound=db_bound,
            loss=loss,
            kappa=kappa,
            verbose=self.verbose,
        )
        assert isinstance(result, np.ndarray)
        return result

    def imperceptible(
        self,
        audio: torch.Tensor | np.ndarray,
        target: str | list[str] | None = None,
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

        Default ``mode="imperceptible"`` is the offline Sec. 4 attack: stage 1
        is a CTC-only L-inf bound search with signed gradient steps; stage 2
        minimizes ``CTC + alpha * l_theta`` (CTC stands in for the paper's
        Lingvo cross-entropy). ``mode="robust"`` / ``"imperceptible_robust"``
        need a :class:`~asr_attacks.rooms.RoomSimulator` via ``rooms=``.

        Targeted mode follows Qin et al. Untargeted (``targeted=False``) is a
        package extension. ``c`` is deprecated and ignored (warns if passed).

        Args:
            audio: Clean waveform.
            target: Target transcript when ``targeted=True``. Optional when
                ``targeted=False`` (then ``label`` or the clean decode is used).
            epsilon: Stage-1 / R1 L-inf start bound (default ≈ 0.061).
            c: Deprecated; ignored with a warning when not ``None``.
            learning_rate1: Stage-1 step size (default ≈ 100/32768).
            learning_rate2: Stage-2 Adam step size (default ≈ 1/32768).
            num_iter1: Stage-1 steps (default 1000).
            num_iter2: Stage-2 steps (default 4000).
            decrease_factor_eps: Bound shrink factor on success.
            num_iter_decrease_eps: Used as ``check_every`` when that is ``None``.
            check_every: Bound-shrink / success-check interval.
            optimizer1: Stage-1 optimizer (``"sgd"`` reproduces Algorithm 1).
            optimizer2: Stage-2 optimizer (default ``"adam"``).
            nested: Hide progress bars when nested.
            early_stop_cw: Early-stop stage 1.
            search_eps_cw: Epsilon search in stage 1.
            alpha: Initial weight on the psychoacoustic penalty (default 0.05).
            sample_rate: Waveform sample rate for the masking threshold.
            targeted: Targeted (default) vs untargeted (extension).
            label: Ground-truth for untargeted attacks.
            mode: ``"imperceptible"``, ``"robust"``, or ``"imperceptible_robust"``.
            rooms: Room simulator required for robust modes.
            robust_delta: Extra bound for R2 / IR stages (≈ 300/32768).
            num_iter_r1: Robust stage R1 iterations.
            num_iter_r2: Robust stage R2 iterations.
            learning_rate_r1: R1 step size.
            learning_rate_r2: R2 step size.
            num_iter_ir1: Imperceptible+robust IR1 iterations.
            num_iter_ir2: Imperceptible+robust IR2 iterations.
            learning_rate_ir1: IR1 Adam step size.
            learning_rate_ir2: IR2 Adam step size.

        Returns:
            CPU NumPy waveform (best successful stage-2 / IR iterate, or the
            stage-1 / robust fallback).
        """
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
            check_every=check_every,
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            nested=nested,
            early_stop_cw=early_stop_cw,
            search_eps_cw=search_eps_cw,
            alpha=alpha,
            sample_rate=sample_rate,
            verbose=self.verbose,
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

    def wer(
        self,
        references: list[str],
        audios: list[np.ndarray],
    ) -> tuple[float, list[tuple[int, int, int]]]:
        """Mean Levenshtein word error rate over a batch.

        ``|`` in transcripts is treated as a space before scoring.

        Args:
            references: Ground-truth transcripts, one per utterance.
            audios: Waveforms decoded with :meth:`decode`.

        Returns:
            Pair of ``(mean_wer, counts)`` where each counts tuple is
            ``(substitutions, insertions, deletions)``.
        """
        scores: list[float] = []
        counts: list[tuple[int, int, int]] = []
        for reference, sample in zip(references, audios, strict=True):
            hypothesis = self.decode(sample)
            scores.append(word_error_rate(reference, hypothesis))
            counts.append(alignment_counts(reference, hypothesis))
        return float(sum(scores) / len(scores)), counts
