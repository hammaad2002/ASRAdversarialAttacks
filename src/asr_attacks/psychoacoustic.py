# Copyright 2018-2024 IBM Corporation and contributors (Adversarial Robustness Toolbox)
# Copyright 2023-2026 Hammad Ali Khan (adaptations for this repository)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This module is derived from ART's PyTorch Imperceptible ASR attack:
# https://github.com/Trusted-AI/adversarial-robustness-toolbox
# art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py
"""Psychoacoustic masking helpers used by the imperceptible ASR attack."""

from __future__ import annotations

import numpy as np
import scipy.signal
import torch


def _stft(
    waveform: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Short-time Fourier transform without padding (``librosa.stft(center=False)``).

    Frames start at ``0, hop_length, 2 * hop_length, ...`` and only complete frames are
    kept, so the result has ``1 + (len - n_fft) // hop_length`` columns. A ``window``
    shorter than ``n_fft`` is zero-padded symmetrically, as librosa does.

    Returns:
        Complex array of shape ``(1 + n_fft // 2, n_frames)``.
    """
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected a mono waveform of shape (samples,), got {signal.shape}")
    if signal.size < n_fft:
        raise ValueError(
            f"Audio has {signal.size} samples but the masking threshold needs at least "
            f"n_fft={n_fft} (about {n_fft / 16000:.2f} s at 16 kHz)"
        )
    if win_length > n_fft:
        raise ValueError(f"win_length={win_length} cannot exceed n_fft={n_fft}")
    padded_window = np.zeros(n_fft, dtype=np.float64)
    left = (n_fft - win_length) // 2
    padded_window[left : left + win_length] = window
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop_length]
    return np.fft.rfft(frames * padded_window, n=n_fft, axis=-1).T


def psd_transform(
    delta: torch.Tensor,
    original_max_psd: np.ndarray | float,
    device: torch.device | str,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
) -> torch.Tensor:
    window = torch.hann_window(win_length, device=device)
    delta_stft = torch.view_as_real(
        torch.stft(
            delta,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            window=window,
            return_complex=True,
        )
    )
    transformed = torch.sqrt(torch.sum(torch.square(delta_stft), -1))
    psd = ((8.0 / 3.0) * transformed / win_length) ** 2
    scale = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), 9.6)
    max_psd = torch.reshape(torch.as_tensor(original_max_psd, device=device), [-1, 1, 1])
    return scale / max_psd * psd.type(torch.float64)


def compute_masking_threshold(
    waveform: np.ndarray,
    win_length: int = 2048,
    hop_length: int = 512,
    n_fft: int = 2048,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, float]:
    window = scipy.signal.get_window("hann", win_length, fftbins=True)
    transformed = _stft(waveform, n_fft, hop_length, win_length, window)
    transformed = transformed * np.sqrt(8.0 / 3.0)
    psd = abs(transformed / win_length)
    original_max_psd = float(np.max(psd * psd))
    with np.errstate(divide="ignore"):
        psd = (20 * np.log10(psd)).clip(min=-200)
    psd = 96 - np.max(psd) + psd

    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

    ath = np.zeros(len(barks), dtype=np.float64) - np.inf
    bark_idx = int(np.argmax(barks > 1))
    ath[bark_idx:] = (
        3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
        + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
        - 12
    )

    theta = []
    for i in range(psd.shape[1]):
        masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]
        if 0 in masker_idx:
            masker_idx = np.delete(masker_idx, 0)
        if len(psd[:, i]) - 1 in masker_idx:
            masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

        barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
        barks_psd[:, 0] = barks[masker_idx]
        barks_psd[:, 1] = 10 * np.log10(
            pow(10, psd[:, i][masker_idx - 1] / 10.0)
            + pow(10, psd[:, i][masker_idx] / 10.0)
            + pow(10, psd[:, i][masker_idx + 1] / 10.0)
        )
        barks_psd[:, 2] = masker_idx

        for j in range(len(masker_idx)):
            if barks_psd.shape[0] <= j + 1:
                break
            while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                quiet_threshold = (
                    3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                    - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                    + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                    - 12
                )
                if barks_psd[j, 1] < quiet_threshold:
                    barks_psd = np.delete(barks_psd, j, axis=0)
                if barks_psd.shape[0] == j + 1:
                    break
                if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                    barks_psd = np.delete(barks_psd, j, axis=0)
                else:
                    barks_psd = np.delete(barks_psd, j + 1, axis=0)
                if barks_psd.shape[0] == j + 1:
                    break

        delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])
        t_s = []
        for m in range(barks_psd.shape[0]):
            d_z = barks - barks_psd[m, 0]
            zero_idx = int(np.argmax(d_z > 0))
            s_f = np.zeros(len(d_z), dtype=np.float64)
            s_f[:zero_idx] = 27 * d_z[:zero_idx]
            s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
            t_s.append(barks_psd[m, 1] + delta[m] + s_f)
        t_s_array = np.array(t_s)
        theta.append(np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0))

    return np.array(theta), original_max_psd
