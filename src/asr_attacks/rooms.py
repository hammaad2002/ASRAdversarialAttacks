"""Room impulse response transforms for robust / imperceptible+robust attacks.

Qin et al. (2019) use the image-source method; this module wraps
``pyroomacoustics`` when available and also accepts user-supplied RIRs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from asr_attacks.backends.base import ASRBackend
from asr_attacks.metrics import attack_succeeded
from asr_attacks.tensors import prepare_audio
from asr_attacks.text import as_transcript

Transform = Callable[[torch.Tensor], torch.Tensor]

# Package defaults — the paper does not publish room-dimension / RT60 ranges.
_DEFAULT_ROOM_DIM_RANGES = ((3.0, 10.0), (3.0, 10.0), (2.5, 4.0))
_DEFAULT_RT60 = (0.2, 0.8)


def _fft_convolve_same_length(audio: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Linear convolution via FFT, truncated to the input length."""
    squeezed = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeezed = True
    if rir.dim() == 1:
        rir = rir.unsqueeze(0)
    n_audio = audio.shape[-1]
    n_rir = rir.shape[-1]
    n_full = n_audio + n_rir - 1
    n_fft = 1 << int(np.ceil(np.log2(max(n_full, 1))))
    audio_f = torch.fft.rfft(audio, n=n_fft, dim=-1)
    rir_f = torch.fft.rfft(rir.to(device=audio.device, dtype=audio.dtype), n=n_fft, dim=-1)
    conv = torch.fft.irfft(audio_f * rir_f, n=n_fft, dim=-1)[..., :n_audio]
    if squeezed:
        return conv.squeeze(0)
    return conv


def _as_rir_array(rir: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    array = np.asarray(
        rir.detach().cpu().numpy() if isinstance(rir, torch.Tensor) else rir,
        dtype=np.float64,
    ).reshape(-1)
    if array.size == 0:
        raise ValueError("RIR must be non-empty")
    return array


class RoomSimulator:
    """Sample differentiable room transforms ``t(x) = fftconv(x, r)[:len(x)]``."""

    def __init__(
        self,
        num_rooms: int = 1000,
        seed: int | None = None,
        room_dims: Sequence[tuple[float, float]] = _DEFAULT_ROOM_DIM_RANGES,
        rt60: tuple[float, float] = _DEFAULT_RT60,
        sample_rate: int = 16000,
        *,
        _rirs: Sequence[np.ndarray | torch.Tensor | Sequence[float]] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed)
        if _rirs is not None:
            self._rirs = [_as_rir_array(rir) for rir in _rirs]
            if not self._rirs:
                raise ValueError("from_rirs requires a non-empty list of RIRs")
            return
        try:
            import pyroomacoustics as pra
        except ImportError as exc:
            raise ImportError(
                "RoomSimulator generation requires the 'rooms' extra: "
                "pip install asr-attacks[rooms]"
            ) from exc

        self._rirs = []
        for _ in range(num_rooms):
            dims = [float(self._rng.uniform(lo, hi)) for lo, hi in room_dims]
            absorption, max_order = pra.inverse_sabine(
                float(self._rng.uniform(rt60[0], rt60[1])),
                dims,
            )
            room = pra.ShoeBox(
                dims,
                fs=sample_rate,
                materials=pra.Material(absorption),
                max_order=max_order,
            )
            # Source and mic at random interior points, ≥0.5 m from walls.
            src = [float(self._rng.uniform(0.5, d - 0.5)) for d in dims]
            mic = [float(self._rng.uniform(0.5, d - 0.5)) for d in dims]
            room.add_source(src)
            room.add_microphone(mic)
            room.compute_rir()
            rir = np.asarray(room.rir[0][0], dtype=np.float64).reshape(-1)
            peak = np.max(np.abs(rir))
            if peak > 0:
                rir = rir / peak
            self._rirs.append(rir)

    @classmethod
    def from_rirs(
        cls,
        rirs: Sequence[np.ndarray | torch.Tensor | Sequence[float]],
    ) -> RoomSimulator:
        """Build a simulator from user-supplied impulse responses (no pyroomacoustics)."""
        return cls(_rirs=list(rirs))

    @property
    def rirs(self) -> list[np.ndarray]:
        return self._rirs

    def _transform_for(self, rir: np.ndarray) -> Transform:
        rir_cpu = np.asarray(rir, dtype=np.float32)

        def transform(audio: torch.Tensor) -> torch.Tensor:
            rir_t = torch.as_tensor(rir_cpu, device=audio.device, dtype=audio.dtype)
            return _fft_convolve_same_length(audio, rir_t)

        return transform

    def sample(self) -> Transform:
        index = int(self._rng.integers(0, len(self._rirs)))
        return self._transform_for(self._rirs[index])

    def sample_set(self, m: int) -> list[Transform]:
        return [self.sample() for _ in range(m)]


def evaluate_over_rooms(
    backend: ASRBackend,
    audio: torch.Tensor | np.ndarray,
    target: str | list[str],
    rooms: RoomSimulator,
    n: int = 100,
    *,
    targeted: bool = True,
) -> float:
    """Fraction of ``n`` held-out rooms where the attack succeeds."""
    waveform = prepare_audio(audio, backend.device)
    reference = as_transcript(target)
    successes = 0
    for _ in range(n):
        transformed = rooms.sample()(waveform)
        hypothesis = backend.decode(transformed)
        if attack_succeeded(hypothesis, reference, targeted=targeted):
            successes += 1
    return successes / float(n)
