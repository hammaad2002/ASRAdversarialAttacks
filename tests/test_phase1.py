import numpy as np
import pytest
import torch

from asr_attacks.attacker import ASRAttacker
from asr_attacks.attacks.common import ctc_loss, resolve_target
from asr_attacks.attacks.cw import cw
from asr_attacks.attacks.fgsm import fgsm
from asr_attacks.attacks.iterative import bim, pgd
from asr_attacks.compat import ASRAttacks
from asr_attacks.tensors import (
    _parse_norm,
    normalized_step,
    project,
    project_linf,
    random_ball,
)
from tests.helpers import LABELS, TinyCTC

NORMS = ["inf", np.inf, float("inf"), 2, 1]
LABEL = "CAB"


def _lp(delta: np.ndarray | torch.Tensor, norm) -> float:
    p = _parse_norm(norm)
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    if p == float("inf"):
        return float(np.abs(values).max())
    return float(np.linalg.norm(values, ord=p))


def _grad(backend, audio: torch.Tensor, text: str) -> torch.Tensor:
    x = audio.clone().requires_grad_(True)
    ctc_loss(backend.logits(x), backend.encode(text), backend.blank_id).backward()
    return x.grad.detach()


# resolve_target


def test_resolve_target_untargeted_defaults_to_clean_decode(backend, audio):
    assert resolve_target(backend, audio, None, False) == backend.decode(audio)


def test_resolve_target_untargeted_uses_label(backend, audio):
    assert resolve_target(backend, audio, None, False, label=LABEL) == LABEL
    assert resolve_target(backend, audio, None, False, label=["C", "A", "B"]) == LABEL
    assert resolve_target(backend, audio, None, False, label=[LABEL]) == LABEL


def test_resolve_target_targeted(backend, audio):
    assert resolve_target(backend, audio, "AB", True) == "AB"
    with pytest.raises(ValueError, match="target"):
        resolve_target(backend, audio, None, True)
    with pytest.raises(ValueError, match="label"):
        resolve_target(backend, audio, "AB", True, label=LABEL)


# fgsm norms and label


def test_fgsm_inf_step_is_signed_gradient(backend, audio):
    epsilon = 0.01
    adversarial = fgsm(backend, audio, epsilon=epsilon, norm="inf")
    grad = _grad(backend, audio, backend.decode(audio))
    delta = adversarial - audio.numpy()
    assert np.abs(delta).max() <= epsilon + 1e-6
    np.testing.assert_allclose(delta, epsilon * grad.sign().numpy(), atol=1e-6)


@pytest.mark.parametrize("norm", [2, 1])
def test_fgsm_lp_step_has_norm_epsilon(backend, audio, norm):
    epsilon = 0.01
    adversarial = fgsm(backend, audio, epsilon=epsilon, norm=norm)
    grad = _grad(backend, audio, backend.decode(audio))
    delta = adversarial - audio.numpy()
    assert _lp(delta, norm) == pytest.approx(epsilon, rel=1e-3)
    expected = epsilon * (grad / grad.norm(p=norm)).numpy()
    np.testing.assert_allclose(delta, expected, atol=1e-6)


@pytest.mark.parametrize("norm", NORMS)
def test_fgsm_output_stays_in_audio_range(backend, norm):
    loud = torch.full((1, 400), 0.999)
    adversarial = fgsm(backend, loud, epsilon=0.5, norm=norm)
    assert adversarial.max() <= 1.0 and adversarial.min() >= -1.0


def test_fgsm_targeted_steps_against_gradient(backend, audio):
    epsilon = 0.01
    adversarial = fgsm(backend, audio, epsilon=epsilon, targeted=True, target="AB", norm=2)
    grad = _grad(backend, audio, "AB")
    expected = -epsilon * (grad / grad.norm(p=2)).numpy()
    np.testing.assert_allclose(adversarial - audio.numpy(), expected, atol=1e-6)


def test_fgsm_label_changes_perturbation(backend, audio):
    assert backend.decode(audio) != LABEL
    default = fgsm(backend, audio, epsilon=0.01)
    labelled = fgsm(backend, audio, epsilon=0.01, label=LABEL)
    assert not np.allclose(default, labelled)
    grad = _grad(backend, audio, LABEL)
    np.testing.assert_allclose(labelled - audio.numpy(), 0.01 * grad.sign().numpy(), atol=1e-6)


def test_fgsm_label_with_targeted_raises(backend, audio):
    with pytest.raises(ValueError, match="label"):
        fgsm(backend, audio, targeted=True, target="AB", label=LABEL)


@pytest.mark.parametrize("norm", [3, 0, "l2", "2", True, None, -1])
def test_invalid_norm_raises(backend, audio, norm):
    with pytest.raises(ValueError, match="norm"):
        fgsm(backend, audio, epsilon=0.01, norm=norm)
    with pytest.raises(ValueError, match="norm"):
        normalized_step(audio, norm)


# label pass-through for iterative attacks and cw


def test_bim_pgd_cw_label_changes_perturbation(backend, audio):
    kwargs = dict(nested=False, verbose=False)
    for attack, extra in (
        (bim, dict(epsilon=0.02, alpha=0.005, num_iter=3)),
        (pgd, dict(epsilon=0.02, alpha=0.005, num_iter=3, random_start=False)),
        (cw, dict(epsilon=0.02, num_iter=3, early_stop=False)),
    ):
        default = attack(backend, audio, **extra, **kwargs)
        labelled = attack(backend, audio, label=LABEL, **extra, **kwargs)
        assert not np.allclose(default, labelled), attack.__name__
        with pytest.raises(ValueError, match="label"):
            attack(backend, audio, targeted=True, target="AB", label=LABEL, **extra, **kwargs)


def test_bim_early_stop_compares_against_label(backend, audio, monkeypatch):
    calls = []
    original_decode = backend.decode
    monkeypatch.setattr(backend, "decode", lambda x: calls.append(1) or original_decode(x))
    bim(
        backend,
        audio,
        epsilon=0.02,
        alpha=0.005,
        num_iter=20,
        label=LABEL,
        early_stop=True,
        nested=False,
        verbose=False,
    )
    assert len(calls) == 1


def test_wrappers_pass_label_and_norm(backend, audio):
    attacker = ASRAttacker(backend, verbose=False)
    direct = fgsm(backend, audio, epsilon=0.01, label=LABEL, norm=2)
    np.testing.assert_allclose(attacker.fgsm(audio, epsilon=0.01, label=LABEL, norm=2), direct)

    torch.manual_seed(0)
    compat = ASRAttacks(TinyCTC(), "cpu", LABELS)
    np.testing.assert_allclose(
        compat.FGSM_ATTACK(audio, epsilon=0.01, label=LABEL, norm=2), direct, atol=1e-7
    )
    for method in (compat.BIM_ATTACK, compat.PGD_ATTACK):
        with pytest.raises(ValueError, match="label"):
            method(audio, target="AB", targeted=True, label=LABEL, num_iter=1)
    with pytest.raises(ValueError, match="label"):
        compat.CW_ATTACK(audio, target="AB", targeted=True, label=LABEL, num_iter=1)


# tensor helpers


def test_normalized_step_shapes():
    grad = torch.tensor([[3.0, -4.0, 0.0]])
    torch.testing.assert_close(normalized_step(grad, "inf"), torch.tensor([[1.0, -1.0, 0.0]]))
    torch.testing.assert_close(normalized_step(grad, 2), torch.tensor([[0.6, -0.8, 0.0]]))
    torch.testing.assert_close(normalized_step(grad, 1), torch.tensor([[3 / 7, -4 / 7, 0.0]]))
    assert torch.isfinite(normalized_step(torch.zeros(1, 3), 2)).all()


@pytest.mark.parametrize("norm", NORMS)
def test_project_stays_in_ball_and_range(norm):
    torch.manual_seed(0)
    original = torch.rand(1, 500) * 1.8 - 0.9
    adversarial = original + torch.randn(1, 500)
    epsilon = 0.5
    projected = project(adversarial, original, epsilon, norm)
    assert _lp(projected - original, norm) <= epsilon + 1e-5
    assert projected.max() <= 1.0 and projected.min() >= -1.0


@pytest.mark.parametrize("norm", NORMS)
def test_project_is_identity_inside_ball(norm):
    original = torch.zeros(1, 4)
    adversarial = torch.tensor([[0.01, -0.02, 0.0, 0.03]])
    torch.testing.assert_close(project(adversarial, original, 1.0, norm), adversarial)


def test_project_linf_matches_project_inf():
    torch.manual_seed(0)
    original = torch.rand(1, 50) * 2 - 1
    adversarial = original + torch.randn(1, 50)
    torch.testing.assert_close(
        project_linf(adversarial, original, 0.1), project(adversarial, original, 0.1, "inf")
    )


def test_project_l1_known_vector():
    original = torch.zeros(1, 3)
    projected = project(torch.tensor([[0.3, 0.1, -0.2]]), original, 0.2, 1)
    torch.testing.assert_close(projected, torch.tensor([[0.15, 0.0, -0.05]]))


def test_project_l1_matches_brute_force():
    torch.manual_seed(3)
    original = torch.zeros(1, 6, dtype=torch.float64)
    v = torch.randn(1, 6, dtype=torch.float64) * 0.3
    epsilon = 0.25
    projected = project(v, original, epsilon, 1)
    assert projected.abs().sum().item() == pytest.approx(epsilon, abs=1e-9)

    thetas = torch.linspace(0, v.abs().max().item(), 200001, dtype=torch.float64)
    l1 = torch.clamp(v.abs() - thetas[:, None], min=0).sum(dim=1)
    theta = thetas[(l1 - epsilon).abs().argmin()]
    brute = v.sign() * torch.clamp(v.abs() - theta, min=0)
    torch.testing.assert_close(projected, brute, atol=1e-5, rtol=0)

    candidates = torch.randn(20000, 6, dtype=torch.float64)
    candidates *= epsilon / candidates.abs().sum(dim=1, keepdim=True)
    assert (candidates - v).norm(dim=1).min().item() >= (projected - v).norm().item() - 1e-12


def test_project_clips_to_custom_range():
    original = torch.full((1, 5), 0.95)
    projected = project(original + 0.5, original, 0.5, 2, clip_min=-0.5, clip_max=0.97)
    assert projected.max() <= 0.97


@pytest.mark.parametrize("norm", NORMS)
def test_random_ball_within_budget(norm):
    torch.manual_seed(0)
    original = torch.zeros(1, 800)
    epsilon = 0.1
    for _ in range(20):
        delta = random_ball(original, epsilon, norm)
        assert delta.shape == original.shape
        assert _lp(delta, norm) <= epsilon + 1e-6
        assert _lp(delta, norm) > 0


def test_random_ball_invalid_norm_raises():
    with pytest.raises(ValueError, match="norm"):
        random_ball(torch.zeros(1, 4), 0.1, 3)
    with pytest.raises(ValueError, match="norm"):
        project(torch.zeros(1, 4), torch.zeros(1, 4), 0.1, "l1")
