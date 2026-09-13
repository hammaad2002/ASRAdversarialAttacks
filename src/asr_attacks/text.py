from __future__ import annotations

from collections.abc import Sequence


def as_transcript(value: str | Sequence[str]) -> str:
    """Accept a string, a one-element sentence list, or a character list."""
    if isinstance(value, str):
        return value
    if len(value) == 1:
        return str(value[0])
    return "".join(str(part) for part in value)


def display_text(value: str) -> str:
    """Normalize wav2vec-style ``|`` word boundaries into spaces."""
    return " ".join(value.replace("|", " ").split())
