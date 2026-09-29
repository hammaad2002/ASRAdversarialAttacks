from __future__ import annotations

import numpy as np
import torch


def prepare_audio(audio: torch.Tensor | np.ndarray, device: torch.device | str) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    tensor = audio.to(device=device, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError(
            "Pass one utterance at a time with shape (samples,) or (1, samples). "
            f"Got {tuple(tensor.shape)}. Squeeze extra channel axes; do not stack files."
        )
    return tensor


def to_numpy(audio: torch.Tensor) -> np.ndarray:
    return audio.detach().cpu().numpy()


def _parse_norm(norm: float | str) -> float:
    """Return ``inf``, ``2.0`` or ``1.0`` for a supported norm, else raise ``ValueError``."""
    if isinstance(norm, str):
        if norm.lower() == "inf":
            return float("inf")
    elif not isinstance(norm, bool) and isinstance(norm, (int, float, np.integer, np.floating)):
        if norm in (float("inf"), 2, 1):
            return float(norm)
    raise ValueError(f"norm must be 'inf', np.inf, 2 or 1; got {norm!r}")


def normalized_step(grad: torch.Tensor, norm: float | str) -> torch.Tensor:
    """Unit steepest-ascent direction: ``sign(g)`` for inf, ``g / ||g||_p`` for 2 and 1."""
    p = _parse_norm(norm)
    if p == float("inf"):
        return grad.sign()
    return grad / grad.norm(p=p).clamp_min(1e-12)


def _project_l1_ball(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Exact Euclidean projection onto the L1 ball (Duchi et al., 2008, sort-based)."""
    flat = delta.reshape(-1)
    magnitude = flat.abs()
    if magnitude.sum() <= epsilon:
        return delta
    sorted_mag, _ = torch.sort(magnitude, descending=True)
    cumulative = torch.cumsum(sorted_mag, dim=0)
    ranks = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=flat.dtype)
    support = sorted_mag - (cumulative - epsilon) / ranks > 0
    rho = int(torch.nonzero(support).max().item())
    theta = (cumulative[rho] - epsilon) / (rho + 1)
    projected = flat.sign() * torch.clamp(magnitude - theta, min=0.0)
    return projected.reshape(delta.shape)


def project(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    norm: float | str,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Project onto the L-``norm`` ball of radius ``epsilon`` around ``original``, then clip."""
    p = _parse_norm(norm)
    delta = adversarial - original
    if p == float("inf"):
        delta = torch.clamp(delta, -epsilon, epsilon)
    elif p == 2:
        scale = torch.clamp(epsilon / delta.norm(p=2).clamp_min(1e-12), max=1.0)
        delta = delta * scale
    else:
        delta = _project_l1_ball(delta, epsilon)
    return torch.clamp(original + delta, clip_min, clip_max)


def project_linf(
    adversarial: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Project onto the L-inf ball around ``original``, then clip to valid audio range."""
    return project(adversarial, original, epsilon, "inf", clip_min=clip_min, clip_max=clip_max)


def random_ball(original: torch.Tensor, epsilon: float, norm: float | str) -> torch.Tensor:
    """Sample a random perturbation inside the L-``norm`` ball of radius ``epsilon``.

    inf: uniform in ``[-epsilon, epsilon]``. 2: Gaussian direction scaled by
    ``epsilon * u**(1/d)``. 1: random signs times exponentials, normalized to
    unit L1 norm, scaled by ``epsilon * u**(1/d)``.
    """
    p = _parse_norm(norm)
    if p == float("inf"):
        return torch.empty_like(original).uniform_(-epsilon, epsilon)
    dims = original.numel()
    radius = epsilon * torch.rand((), device=original.device, dtype=original.dtype) ** (1.0 / dims)
    if p == 2:
        direction = torch.randn_like(original)
    else:
        signs = torch.randint_like(original, 0, 2) * 2 - 1
        direction = signs * torch.empty_like(original).exponential_()
    return direction / direction.norm(p=p).clamp_min(1e-12) * radius
