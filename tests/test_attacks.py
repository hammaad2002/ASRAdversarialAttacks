import numpy as np

from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.iterative import bim, pgd
from asr_attacks.compat import ASRAttacks
from tests.helpers import LABELS, TinyCTC


def test_fgsm_returns_cpu_numpy_and_changes_audio(backend, audio):
    original = audio.clone()
    adversarial = fgsm(backend, audio, epsilon=0.05, targeted=False)
    assert isinstance(adversarial, np.ndarray)
    assert adversarial.shape == original.shape
    assert not np.allclose(adversarial, original.numpy(), atol=1e-8)


def test_bim_respects_epsilon_ball(backend, audio):
    epsilon = 0.04
    adversarial = bim(
        backend,
        audio,
        epsilon=epsilon,
        alpha=0.02,
        num_iter=12,
        nested=False,
        targeted=False,
        early_stop=False,
        verbose=False,
    )
    delta = np.abs(adversarial - audio.numpy())
    assert delta.max() <= epsilon + 1e-5
    assert adversarial.max() <= 1.0 + 1e-5
    assert adversarial.min() >= -1.0 - 1e-5


def test_pgd_random_start_stays_in_ball(backend, audio):
    epsilon = 0.03
    adversarial = pgd(
        backend,
        audio,
        epsilon=epsilon,
        alpha=0.01,
        num_iter=8,
        nested=False,
        random_start=True,
        verbose=False,
    )
    assert np.abs(adversarial - audio.numpy()).max() <= epsilon + 1e-5


def test_compat_class_matches_old_method_names(audio):
    model = TinyCTC()
    attacks = ASRAttacks(model, "cpu", LABELS)
    result = attacks.FGSM_ATTACK(audio, epsilon=0.02, targeted=False)
    assert isinstance(result, np.ndarray)
    transcript = attacks.INFER(audio)
    assert isinstance(transcript, str)
    wer, details = attacks.wer_compute([transcript.replace("|", " ")], [audio.numpy()])
    assert 0.0 <= wer <= 1.0
    assert len(details) == 1
