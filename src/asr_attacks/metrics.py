from __future__ import annotations

import jiwer

from asr_attacks.text import display_text


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
