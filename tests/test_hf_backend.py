"""HuggingFaceCTCBackend feeds the model the same normalized audio the processor would."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from asr_attacks.attacks.iterative import bim
from asr_attacks.backends.huggingface import HuggingFaceCTCBackend, zero_mean_unit_variance

SR = 16000


class _Recorder(nn.Module):
    """Stands in for ``AutoModelForCTC``: remembers its input, returns differentiable logits."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 3)
        self.seen: torch.Tensor | None = None

    def forward(self, values: torch.Tensor):
        self.seen = values
        return SimpleNamespace(logits=self.proj(values.unsqueeze(-1)))


def _stub_backend(processor) -> HuggingFaceCTCBackend:
    """Build the backend without importing transformers (only ``logits`` is exercised)."""
    backend = object.__new__(HuggingFaceCTCBackend)
    backend.device = torch.device("cpu")
    backend.model = _Recorder()
    backend.processor = processor
    backend.tokenizer = None
    backend.blank_id = 0
    return backend


def _processor(do_normalize: bool):
    return SimpleNamespace(feature_extractor=SimpleNamespace(do_normalize=do_normalize))


@pytest.fixture
def waveform() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, 4000) * 0.1 + 0.05  # non-zero mean, non-unit variance


def test_normalize_true_feeds_zero_mean_unit_variance(waveform):
    backend = _stub_backend(_processor(do_normalize=True))
    backend.logits(waveform)

    seen = backend.model.seen
    assert seen is not None
    reference = (waveform.numpy() - waveform.numpy().mean()) / np.sqrt(
        waveform.numpy().var() + 1e-7
    )
    np.testing.assert_allclose(seen.detach().numpy(), reference, atol=1e-5)
    assert abs(seen.mean().item()) < 1e-5
    assert seen.var(correction=0).item() == pytest.approx(1.0, abs=1e-3)


def test_normalize_false_leaves_the_audio_untouched(waveform):
    backend = _stub_backend(_processor(do_normalize=False))
    backend.logits(waveform)
    torch.testing.assert_close(backend.model.seen, waveform)


@pytest.mark.parametrize(
    "processor",
    [None, SimpleNamespace(), SimpleNamespace(feature_extractor=SimpleNamespace())],
    ids=["none", "empty", "no-flag"],
)
def test_missing_flag_means_no_normalization(processor, waveform):
    backend = _stub_backend(processor)
    backend.logits(waveform)
    torch.testing.assert_close(backend.model.seen, waveform)


def test_processor_that_is_a_bare_feature_extractor(waveform):
    backend = _stub_backend(SimpleNamespace(do_normalize=True))
    backend.logits(waveform)
    assert backend.model.seen is not None
    assert backend.model.seen.mean().abs().item() < 1e-5


def test_normalization_is_differentiable_and_scale_invariant(waveform):
    backend = _stub_backend(_processor(do_normalize=True))
    weights = torch.randn(1, 4000, 3)

    def loss_and_grad(scale: float):
        x = (waveform * scale).clone().requires_grad_(True)
        loss = (backend.logits(x) * weights).sum()
        loss.backward()
        assert x.grad is not None
        return loss.detach(), x.grad

    loss_1, grad_1 = loss_and_grad(1.0)
    loss_3, grad_3 = loss_and_grad(3.0)

    assert torch.isfinite(grad_1).all()
    assert grad_1.abs().sum() > 0
    # The model only sees normalized audio, so rescaling the waveform cannot change the loss...
    torch.testing.assert_close(loss_1, loss_3, atol=1e-3, rtol=1e-4)
    # ...and the gradient through the normalization is orthogonal to the constant and the
    # signal itself (shift and scale invariance).
    assert grad_1.sum().abs().item() < 1e-3 * grad_1.abs().sum().item()
    assert (grad_1 * waveform).sum().abs().item() < 1e-3 * (grad_1 * waveform).abs().sum().item()
    torch.testing.assert_close(grad_1, grad_3 * 3.0, atol=1e-4, rtol=1e-3)


def test_unnormalized_backend_is_not_scale_invariant(waveform):
    backend = _stub_backend(_processor(do_normalize=False))
    assert not torch.allclose(backend.logits(waveform), backend.logits(waveform * 3.0))


def test_zero_mean_unit_variance_handles_batches_and_constant_audio():
    batch = torch.stack([torch.linspace(-1, 1, 100), 5.0 * torch.linspace(0, 1, 100) + 2.0])
    normalized = zero_mean_unit_variance(batch)
    torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        normalized.var(dim=-1, correction=0), torch.ones(2), atol=1e-4, rtol=1e-4
    )
    constant = zero_mean_unit_variance(torch.full((1, 50), 0.3))
    assert torch.isfinite(constant).all()
    assert constant.abs().max() < 1e-3


# --- Against the real Hugging Face classes (needs the ``hf`` extra) ------------------------


def _hf():
    return pytest.importorskip("transformers")


def _hf_models():
    """``transformers`` with the model/processor classes importable.

    Those classes lazily import scikit-learn and pandas. A machine whose pandas was built
    against another NumPy ABI raises "binary incompatibility" there; that is a broken
    environment, not a bug in this package, so only that case skips.
    """
    transformers = _hf()
    try:
        transformers.Wav2Vec2ForCTC  # noqa: B018 - triggers the lazy import
        transformers.Wav2Vec2Processor  # noqa: B018
    except ValueError as exc:
        if "binary incompatibility" in str(exc):
            pytest.skip(f"transformers cannot import in this environment: {exc}")
        raise
    return transformers


@pytest.mark.parametrize("scale", [1.0, 0.1, 1e-3])
@pytest.mark.parametrize("samples", [1600, 8000, 16001])
def test_matches_wav2vec2_feature_extractor(scale, samples):
    transformers = _hf()
    extractor = transformers.Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    rng = np.random.default_rng(samples)
    signal = (rng.standard_normal(samples) * scale + 0.2 * scale).astype(np.float32)

    expected = np.asarray(extractor(signal, sampling_rate=SR).input_values[0])
    got = zero_mean_unit_variance(torch.from_numpy(signal)[None])[0].numpy()

    # 1e-3 * scale is small enough that the 1e-7 variance floor matters: this pins it down.
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.fixture
def tiny_hf(tmp_path):
    """A random, offline Wav2Vec2ForCTC plus a matching processor and tokenizer."""
    transformers = _hf_models()
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6, "C": 7}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))
    tokenizer = transformers.Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|"
    )

    def build(do_normalize: bool):
        extractor = transformers.Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SR,
            padding_value=0.0,
            do_normalize=do_normalize,
            return_attention_mask=False,
        )
        processor = transformers.Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
        config = transformers.Wav2Vec2Config(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(16, 16),
            conv_stride=(5, 2),
            conv_kernel=(10, 3),
            num_feat_extract_layers=2,
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            pad_token_id=0,
            feat_extract_norm="group",
            mask_time_prob=0.0,
            layerdrop=0.0,
        )
        torch.manual_seed(0)
        model = transformers.Wav2Vec2ForCTC(config)
        backend = HuggingFaceCTCBackend(model, "cpu", processor=processor, tokenizer=tokenizer)
        return backend, extractor

    return build


def test_backend_logits_equal_processor_then_model(tiny_hf):
    backend, extractor = tiny_hf(do_normalize=True)
    torch.manual_seed(3)
    audio = torch.randn(1, 8000) * 0.05 + 0.01

    processed = torch.as_tensor(
        np.asarray(extractor(audio[0].numpy(), sampling_rate=SR).input_values), dtype=torch.float32
    )
    with torch.no_grad():
        expected = backend.model(processed).logits
        got = backend.logits(audio)

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_real_model_sees_normalized_audio_only_when_the_processor_asks(tiny_hf):
    audio = torch.randn(1, 8000) * 0.05
    normalizing, _ = tiny_hf(do_normalize=True)
    plain, _ = tiny_hf(do_normalize=False)

    with torch.no_grad():
        torch.testing.assert_close(
            normalizing.logits(audio), normalizing.logits(4.0 * audio), atol=1e-4, rtol=1e-4
        )
        assert not torch.allclose(plain.logits(audio), plain.logits(4.0 * audio), atol=1e-4)


def test_bim_runs_through_normalization_on_a_real_model(tiny_hf):
    backend, _ = tiny_hf(do_normalize=True)
    torch.manual_seed(4)
    audio = torch.randn(1, 8000) * 0.05

    adversarial = bim(
        backend, audio, epsilon=0.01, num_iter=3, targeted=True, target="AB", verbose=False
    )

    assert adversarial.shape == tuple(audio.shape)
    assert np.isfinite(adversarial).all()
    assert np.abs(adversarial - audio.numpy()).max() <= 0.01 + 1e-6
    assert not np.allclose(adversarial, audio.numpy())
