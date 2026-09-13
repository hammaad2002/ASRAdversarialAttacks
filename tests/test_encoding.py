import pytest
import torch

from asr_attacks.backends.module import CTCModuleBackend
from tests.helpers import TinyCTC


def test_encode_uses_constructor_labels_not_wav2vec_charset():
    labels = ["-", "Q", "#"]
    backend = CTCModuleBackend(TinyCTC(vocab=3), labels=labels, device="cpu")
    ids = backend.encode("Q#Q")
    assert ids.tolist() == [1, 2, 1]


def test_encode_rejects_unknown_characters():
    labels = ["-", "A"]
    backend = CTCModuleBackend(TinyCTC(vocab=2), labels=labels, device="cpu")
    with pytest.raises(ValueError, match="not in the model vocabulary"):
        backend.encode("Z")


def test_spaces_become_word_delimiter_when_present():
    labels = ["-", "A", "|"]
    backend = CTCModuleBackend(TinyCTC(vocab=3), labels=labels, device="cpu")
    ids = backend.encode("A A")
    assert ids.tolist() == [1, 2, 1]


def test_greedy_decode_runs_without_grad(backend, audio):
    torch.set_grad_enabled(True)
    text = backend.decode(audio)
    assert isinstance(text, str)
    assert not text or all(char in backend.labels or char == "" for char in text)
