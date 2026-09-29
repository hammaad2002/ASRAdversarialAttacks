from __future__ import annotations

import jiwer
import numpy as np
import torch

from asr_attacks.text import display_text


def db_distortion(
    original: torch.Tensor | np.ndarray,
    adversarial: torch.Tensor | np.ndarray,
) -> float:
    """Relative perturbation loudness used by Carlini & Wagner (audio), in dB.

    ``20 * log10(max|delta|) - 20 * log10(max|x|)``.
    """
    x = np.asarray(
        original.detach().cpu().numpy() if isinstance(original, torch.Tensor) else original,
        dtype=np.float64,
    )
    adv = np.asarray(
        adversarial.detach().cpu().numpy()
        if isinstance(adversarial, torch.Tensor)
        else adversarial,
        dtype=np.float64,
    )
    max_x = float(np.max(np.abs(x)))
    max_delta = float(np.max(np.abs(adv - x)))
    if max_x <= 0.0:
        raise ValueError("original waveform must have non-zero peak amplitude")
    if max_delta <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(max_delta) - 20.0 * np.log10(max_x))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein word error rate after normalizing ``|`` to spaces."""
    ref = display_text(reference)
    hyp = display_text(hypothesis)
    if ref == "" and hyp == "":
        return 0.0
    if ref == "":
        return 1.0
    return float(jiwer.wer(ref, hyp))


def alignment_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return ``(substitutions, insertions, deletions)``."""
    ref = display_text(reference)
    hyp = display_text(hypothesis)
    if ref == "" and hyp == "":
        return 0, 0, 0
    if ref == "":
        return 0, len(hyp.split()), 0
    result = jiwer.process_words(ref, hyp)
    return int(result.substitutions), int(result.insertions), int(result.deletions)


def attack_succeeded(hypothesis: str, target: str, *, targeted: bool) -> bool:
    if targeted:
        return word_error_rate(target, hypothesis) == 0.0
    return display_text(hypothesis) != display_text(target)
