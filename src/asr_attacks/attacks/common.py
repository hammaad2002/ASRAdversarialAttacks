from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.text import as_transcript


def ctc_loss(logits: torch.Tensor, target_ids: torch.Tensor, blank_id: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    batch, time, _ = logits.shape
    input_lengths = torch.full((batch,), time, dtype=torch.long, device=logits.device)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
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
    )


def resolve_target(backend: ASRBackend, audio: torch.Tensor, target, targeted: bool) -> str:
    if targeted:
        if target is None:
            raise ValueError("A target transcription is required for a targeted attack")
        return as_transcript(target)
    return backend.decode(audio)


def iteration_bar(num_iter: int, nested: bool, desc: str | None = None):
    return tqdm(range(num_iter), leave=not nested, desc=desc)


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
