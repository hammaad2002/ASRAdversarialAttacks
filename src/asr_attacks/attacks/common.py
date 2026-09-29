from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.text import as_transcript


def min_ctc_frames(target_ids: torch.Tensor) -> int:
    """Fewest logit frames that can emit ``target_ids`` under CTC.

    Every token needs a frame, and two identical neighbours also need a blank between
    them, so the minimum is ``len(target) + number of adjacent repeats`` (worst row of
    a batch).
    """
    ids = target_ids.reshape(1, -1) if target_ids.dim() == 1 else target_ids
    if ids.shape[1] == 0:
        return 0
    repeats = (ids[:, 1:] == ids[:, :-1]).sum(dim=1)
    return int((ids.shape[1] + repeats).max().item())


def ctc_loss(logits: torch.Tensor, target_ids: torch.Tensor, blank_id: int) -> torch.Tensor:
    """Mean CTC loss of greedy-decodable ``logits`` against ``target_ids``.

    Raises:
        ValueError: If the audio is too short to spell the target. CTC then has no valid
            alignment, the loss is infinite, and gradient attacks would emit NaN audio.
    """
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    batch, time, _ = logits.shape
    input_lengths = torch.full((batch,), time, dtype=torch.long, device=logits.device)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
    needed = min_ctc_frames(target_ids)
    if needed > time:
        raise ValueError(
            f"The target needs at least {needed} logit frames ({target_ids.shape[1]} tokens "
            f"plus one blank between each pair of repeated tokens) but the model produced "
            f"only {time} for this audio. Use longer audio or a shorter target."
        )
    target_lengths = torch.full(
        (target_ids.shape[0],),
        target_ids.shape[1],
        dtype=torch.long,
        device=logits.device,
    )
    return F.ctc_loss(
        log_probs,
        target_ids,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def resolve_target(
    backend: ASRBackend,
    audio: torch.Tensor,
    target,
    targeted: bool,
    label=None,
) -> str:
    """Return the reference transcript used by the loss and the success check.

    Targeted: ``target``. Untargeted: ``label`` if given, else the greedy decode
    of ``audio``.
    """
    if targeted:
        if label is not None:
            raise ValueError("label is only used for untargeted attacks; pass target instead")
        if target is None:
            raise ValueError("A target transcription is required for a targeted attack")
        return as_transcript(target)
    if label is not None:
        return as_transcript(label)
    return backend.decode(audio)


def iteration_bar(num_iter: int, nested: bool, desc: str | None = None, verbose: bool = True):
    """Progress bar over ``range(num_iter)``; ``verbose=False`` draws nothing."""
    return tqdm(range(num_iter), leave=not nested, desc=desc, disable=not verbose)


def maybe_early_stop(
    backend: ASRBackend,
    audio: torch.Tensor,
    target_text: str,
    *,
    targeted: bool,
    early_stop: bool,
    verbose: bool,
    success_message: str,
) -> bool:
    if not early_stop:
        return False
    hypothesis = backend.decode(audio)
    if attack_succeeded(hypothesis, target_text, targeted=targeted):
        if verbose:
            print(success_message)
        return True
    return False
