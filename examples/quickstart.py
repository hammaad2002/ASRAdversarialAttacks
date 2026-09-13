"""Minimal CPU example of the public API.

This script does not download wav2vec2. It uses a tiny convolutional CTC stand-in
so the attack plumbing can be exercised without model weights.
"""

from __future__ import annotations

import torch
from torch import nn

from asr_attacks import ASRAttacker, CTCModuleBackend


class TinyCTC(nn.Module):
    def __init__(self, vocab: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, vocab, kernel_size=3, padding=1)

    def forward(self, audio: torch.Tensor):
        return self.conv(audio.unsqueeze(1)).transpose(1, 2), None


def main() -> None:
    labels = ["-", "A", "B", "C", "|"]
    backend = CTCModuleBackend(TinyCTC(), labels=labels, device="cpu")
    attacker = ASRAttacker(backend, verbose=False)
    audio = torch.randn(1, 1600) * 0.05
    adversarial = attacker.fgsm(audio, epsilon=0.02, targeted=False)
    print("clean:", attacker.decode(audio))
    print("adv:", attacker.decode(adversarial))
    print("linf:", float((adversarial - audio.numpy()).max()))


if __name__ == "__main__":
    main()
