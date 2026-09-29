import numpy as np
import pytest
import torch

from asr_attacks.metrics import alignment_counts, word_error_rate
from asr_attacks.tensors import prepare_audio, project_linf, to_numpy
from asr_attacks.text import as_transcript, display_text


def test_as_transcript_accepts_string_and_char_list():
    assert as_transcript("HELLO") == "HELLO"
    assert as_transcript(["HELLO"]) == "HELLO"
    assert as_transcript(["H", "E", "L"]) == "HEL"


def test_display_text_normalizes_wav2vec_boundaries():
    assert display_text("THE|CAT") == "THE CAT"


def test_word_error_rate_is_levenshtein():
    assert word_error_rate("the cat", "the cat") == 0.0
    assert word_error_rate("the cat", "the bat") == 0.5
    assert alignment_counts("the cat", "the bat") == (1, 0, 0)


def test_prepare_audio_accepts_1d_and_single_row():
    row = prepare_audio(torch.zeros(8), "cpu")
    assert tuple(row.shape) == (1, 8)
    assert tuple(prepare_audio(torch.zeros(1, 8), "cpu").shape) == (1, 8)


def test_prepare_audio_rejects_stacked_utterances_and_channel_axes():
    with pytest.raises(ValueError, match="one utterance"):
        prepare_audio(torch.zeros(2, 8), "cpu")
    with pytest.raises(ValueError, match="one utterance"):
        prepare_audio(torch.zeros(1, 1, 8), "cpu")


def test_project_linf_stays_inside_epsilon_and_audio_range():
    original = torch.zeros(1, 8)
    adversarial = torch.full((1, 8), 2.0)
    projected = project_linf(adversarial, original, epsilon=0.1)
    assert torch.all(projected <= 0.1 + 1e-6)
    assert torch.all(projected >= -0.1 - 1e-6)
    assert float(projected.max()) <= 1.0


def test_to_numpy_detaches_to_cpu():
    tensor = torch.arange(4, dtype=torch.float32, requires_grad=True)
    array = to_numpy(tensor)
    assert isinstance(array, np.ndarray)
    assert array.tolist() == [0.0, 1.0, 2.0, 3.0]
