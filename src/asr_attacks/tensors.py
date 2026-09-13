from __future__ import annotations

import numpy as np
import torch


def prepare_audio(audio: torch.Tensor | np.ndarray, device: torch.device | str) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    tensor = audio.to(device=device, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            f"Expected audio with shape (samples,) or (batch, samples), got {tuple(tensor.shape)}"
        )
    return tensor


def to_numpy(audio: torch.Tensor) -> np.ndarray:
    return audio.detach().cpu().numpy()


def project_linf(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Project onto the L-inf ball around ``original``, then clip to valid audio range."""
    projected = torch.max(torch.min(adversarial, original + epsilon), original - epsilon)
    return torch.clamp(projected, clip_min, clip_max)
