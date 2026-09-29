from __future__ import annotations

import torch

from asr_attacks.backends.base import ASRBackend
from asr_attacks.tensors import prepare_audio
from asr_attacks.text import as_transcript, display_text


class CTCModuleBackend(ASRBackend):
    """Wrap a PyTorch CTC module such as torchaudio wav2vec2.

    ``labels`` must be the model's vocabulary in index order (``bundle.get_labels()``
    for torchaudio wav2vec2). Encoding uses this vocabulary; it is not hardcoded.
    Logits must be ``(batch, time, vocab)`` with ``vocab == len(labels)``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        labels: list[str] | tuple[str, ...],
        device: torch.device | str | None = None,
        blank_id: int | None = None,
        freeze_model: bool = True,
    ) -> None:
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.labels = list(labels)
        if not self.labels:
            raise ValueError("labels must be a non-empty vocabulary")
        self._char_to_id = {char: index for index, char in enumerate(self.labels)}
        if blank_id is None:
            blank_id = self._char_to_id.get("-", 0)
        self.blank_id = blank_id
        if freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def logits(self, audio: torch.Tensor) -> torch.Tensor:
        prepared = prepare_audio(audio, self.device)
        output = self.model(prepared)
        if isinstance(output, tuple):
            output = output[0]
        elif hasattr(output, "logits"):
            output = output.logits
        if output.dim() != 3:
            raise ValueError(
                f"Model logits must be (batch, time, vocab), got {tuple(output.shape)}"
            )
        vocab = output.shape[-1]
        if vocab != len(self.labels):
            raise ValueError(
                "Model logits last dimension must equal len(labels) "
                f"({vocab} != {len(self.labels)}). "
                "Pass the vocabulary in index order, one label per class."
            )
        return output

    def encode(self, transcript: str | list[str]) -> torch.Tensor:
        text = as_transcript(transcript)
        if " " in text and "|" in self._char_to_id:
            text = text.replace(" ", "|")
        try:
            ids = [self._char_to_id[char] for char in text]
        except KeyError as exc:
            missing = str(exc.args[0])
            raise ValueError(
                f"Character {missing!r} is not in the model vocabulary. "
                "Pass labels from the model (e.g. bundle.get_labels()) and keep "
                "targets inside that charset."
            ) from exc
        if not ids:
            raise ValueError("Encoded CTC target is empty")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def decode(self, audio: torch.Tensor) -> str:
        logits = self.logits(audio)
        predicted = torch.argmax(logits[0], dim=-1)
        predicted = torch.unique_consecutive(predicted)
        chars = [self.labels[int(index)] for index in predicted if int(index) != self.blank_id]
        return "".join(chars)

    def silence_ids(self) -> list[int]:
        ids = [self.blank_id]
        for token in ("|", " "):
            if token in self._char_to_id:
                token_id = self._char_to_id[token]
                if token_id not in ids:
                    ids.append(token_id)
                break
        return ids

    def decode_display(self, audio: torch.Tensor) -> str:
        return display_text(self.decode(audio))
