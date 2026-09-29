"""Masking-threshold code runs for real (no librosa, no mocks)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import scipy.signal
import torch

from asr_attacks.attacks.cw import imperceptible
from asr_attacks.psychoacoustic import _stft, compute_masking_threshold, psd_transform

N_FFT = 2048
HOP = 512
SR = 16000


def _tone(samples: int, freq: float = 440.0, noise: float = 0.001, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / SR
    return 0.5 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(samples)


def test_stft_matches_scipy_reference():
    signal = np.random.default_rng(3).standard_normal(N_FFT + HOP * 9 + 77)
    window = scipy.signal.get_window("hann", N_FFT, fftbins=True)

    ours = _stft(signal, N_FFT, HOP, N_FFT, window)

    # scipy scales the "spectrum" STFT by 1 / sum(window); undo it to compare raw DFTs.
    freqs, _, reference = scipy.signal.stft(
        signal,
        fs=SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        detrend=False,
        boundary=None,
        padded=False,
        scaling="spectrum",
    )
    reference = reference * window.sum()

    assert ours.shape == reference.shape == (N_FFT // 2 + 1, 1 + (signal.size - N_FFT) // HOP)
    np.testing.assert_allclose(ours, reference, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.linspace(0.0, SR / 2.0, N_FFT // 2 + 1), freqs)


def test_stft_pads_short_window_symmetrically():
    signal = np.random.default_rng(4).standard_normal(N_FFT + HOP * 2)
    window = scipy.signal.get_window("hann", 1024, fftbins=True)
    ours = _stft(signal, N_FFT, HOP, 1024, window)

    padded = np.zeros(N_FFT)
    padded[512:1536] = window
    for frame in range(ours.shape[1]):
        chunk = signal[frame * HOP : frame * HOP + N_FFT]
        np.testing.assert_allclose(ours[:, frame], np.fft.rfft(chunk * padded), atol=1e-9)


def test_stft_rejects_short_or_multichannel_audio():
    window = scipy.signal.get_window("hann", N_FFT, fftbins=True)
    with pytest.raises(ValueError, match="at least n_fft"):
        _stft(np.zeros(N_FFT - 1), N_FFT, HOP, N_FFT, window)
    with pytest.raises(ValueError, match="mono"):
        _stft(np.zeros((2, N_FFT * 2)), N_FFT, HOP, N_FFT, window)


def test_masking_threshold_is_finite_and_follows_the_masker():
    samples = N_FFT + HOP * 12
    theta, max_psd = compute_masking_threshold(_tone(samples, 440.0), sample_rate=SR)

    assert theta.shape == (13, N_FFT // 2 + 1)
    assert np.isfinite(theta).all()
    assert (theta > 0).all()
    assert max_psd > 0

    # Bin spacing is 8 kHz / 1024 = 7.8 Hz, so 440 Hz sits near bin 56. A loud tone must
    # mask its neighbourhood far more than the quiet top of the band.
    near_masker = theta[:, 50:62].max(axis=1)
    top_of_band = theta[:, 900]
    assert (near_masker > 100 * top_of_band).all()


def test_masking_threshold_needs_enough_audio():
    with pytest.raises(ValueError, match="at least n_fft"):
        compute_masking_threshold(_tone(N_FFT - 1), sample_rate=SR)


def test_psd_transform_is_differentiable_and_shaped():
    samples = N_FFT + HOP * 12
    delta = torch.from_numpy(_tone(samples, 1000.0, 0.01)).float().requires_grad_(True)
    _, max_psd = compute_masking_threshold(_tone(samples), sample_rate=SR)

    psd = psd_transform(delta, max_psd, "cpu")

    assert psd.shape == (1, N_FFT // 2 + 1, 13)
    assert psd.dtype == torch.float64
    assert torch.isfinite(psd).all()
    psd.sum().backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_package_imports_without_librosa():
    code = (
        "import sys; sys.modules['librosa'] = None; sys.modules['numba'] = None;"
        "import asr_attacks, asr_attacks.psychoacoustic as p, numpy as np;"
        "theta, _ = p.compute_masking_threshold(np.random.randn(4096) * 0.1);"
        "assert np.isfinite(theta).all()"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_imperceptible_stage_two_with_real_masking(backend):
    torch.manual_seed(5)
    audio = torch.randn(1, N_FFT * 3) * 0.05

    out = imperceptible(
        backend,
        audio,
        "AB",
        epsilon=0.05,
        num_iter1=2,
        num_iter2=3,
        learning_rate1=1e-2,
        learning_rate2=1e-3,
        early_stop_cw=False,
        search_eps_cw=False,
        nested=False,
        verbose=False,
        sample_rate=SR,
    )

    assert out.shape == audio.numpy().shape
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0 + 1e-5
