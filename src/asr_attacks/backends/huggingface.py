from __future__ import annotations

import torch

from asr_attacks.backends.base import ASRBackend
from asr_attacks.tensors import prepare_audio
from asr_attacks.text import as_transcript


class HuggingFaceCTCBackend(ASRBackend):
    """Wrap a Hugging Face ``AutoModelForCTC`` checkpoint.

    Install the optional extra: ``pip install asr-attacks[hf]``.
    """

    def __init__(
        self,
        model: str | torch.nn.Module,
        device: torch.device | str = "cpu",
        processor=None,
        tokenizer=None,
        freeze_model: bool = True,
    ) -> None:
        try:
            from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Hugging Face support requires the 'hf' extra: pip install asr-attacks[hf]"
            ) from exc

        self.device = torch.device(device)
        if isinstance(model, str):
            self.model = AutoModelForCTC.from_pretrained(model).to(self.device)
            self.processor = processor or AutoProcessor.from_pretrained(model)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        else:
            if processor is None or tokenizer is None:
                raise ValueError(
                    "Pass processor and tokenizer when wrapping an already-loaded model"
                )
            self.model = model.to(self.device)
            self.processor = processor
            self.tokenizer = tokenizer
        self.model.eval()
        pad_id = getattr(self.model.config, "pad_token_id", None)
        self.blank_id = int(pad_id) if pad_id is not None else 0
        if freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def logits(self, audio: torch.Tensor) -> torch.Tensor:
        prepared = prepare_audio(audio, self.device)
        return self.model(prepared).logits

    def encode(self, transcript: str | list[str]) -> torch.Tensor:
        text = as_transcript(transcript)
        ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        ids = ids.to(self.device).long().view(-1)
        if ids.numel() == 0:
            raise ValueError("Encoded CTC target is empty")
        return ids

    @torch.no_grad()
    def decode(self, audio: torch.Tensor) -> str:
        logits = self.logits(audio)
        predicted = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted)[0]

    def silence_ids(self) -> list[int]:
        ids = [self.blank_id]
        delim = getattr(self.tokenizer, "word_delimiter_token_id", None)
        if delim is not None:
            token_id = int(delim)
            if token_id not in ids:
                ids.append(token_id)
        return ids
