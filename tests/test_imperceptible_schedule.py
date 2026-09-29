"""Stage-2 schedules of Qin et al. (Sec. 4).

* the psychoacoustic weight ``alpha`` grows by 1.2 every 20 steps while the attack succeeds
  and shrinks by 0.8 every 50 steps while it does not;
* the learning rate is divided by 10 after 3000 steps.

The loss is replaced by ``alpha * sum(x)`` so that the gradient the optimiser sees *is*
``alpha``, and success is scripted, which isolates the schedule from the toy model.
"""

from __future__ import annotations

import sys

import pytest
import torch

from asr_attacks.attacks.imperceptible import _imperceptible_stage2

module = sys.modules["asr_attacks.attacks.imperceptible"]

BASE_LR = 1e-3


@pytest.fixture
def trace(monkeypatch):
    seen: dict = {"alpha": [], "lr": [], "success": True}

    monkeypatch.setattr(module, "_psychoacoustic_loss", lambda adversarial, *a: adversarial.sum())
    monkeypatch.setattr(module, "ctc_loss", lambda logits, *a, **k: logits.sum() * 0.0)
    monkeypatch.setattr(module, "attack_succeeded", lambda *a, **k: seen["success"])

    real = module._optimizer_for

    def recording_optimizer(name, parameters, learning_rate):
        optimizer = real(name, parameters, learning_rate)
        real_step = optimizer.step

        def step(*args, **kwargs):
            seen["alpha"].append(float(parameters[0].grad.flatten()[0]))
            seen["lr"].append(optimizer.param_groups[0]["lr"])
            return real_step(*args, **kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(module, "_optimizer_for", recording_optimizer)
    return seen


def _stage2(backend, long_audio, num_iter2):
    return _imperceptible_stage2(
        backend,
        long_audio,
        long_audio,
        "AB",
        targeted=True,
        learning_rate2=BASE_LR,
        num_iter2=num_iter2,
        optimizer2="sgd",
        alpha=1.0,
        sample_rate=16000,
        nested=False,
        verbose=False,
        ctc_ref_stage1=None,
    )


def test_alpha_grows_by_1_2_every_20_steps_while_the_attack_succeeds(backend, long_audio, trace):
    _stage2(backend, long_audio, num_iter2=60)

    alpha = trace["alpha"]
    assert alpha[0] == pytest.approx(1.0)
    assert alpha[19] == pytest.approx(1.0)
    assert alpha[20] == pytest.approx(1.2)
    assert alpha[39] == pytest.approx(1.2)
    assert alpha[40] == pytest.approx(1.44)
    # Step 49 also checks for failure, but a success must not shrink alpha.
    assert alpha[50] == pytest.approx(1.44)
    assert alpha[59] == pytest.approx(1.44)


def test_alpha_shrinks_by_0_8_every_50_steps_while_the_attack_fails(backend, long_audio, trace):
    trace["success"] = False

    _stage2(backend, long_audio, num_iter2=150)

    alpha = trace["alpha"]
    assert alpha[49] == pytest.approx(1.0)
    assert alpha[50] == pytest.approx(0.8)
    assert alpha[99] == pytest.approx(0.8)
    assert alpha[100] == pytest.approx(0.64)


def test_learning_rate_is_divided_by_ten_after_3000_steps(backend, long_audio, trace):
    _stage2(backend, long_audio, num_iter2=3005)

    lr = trace["lr"]
    assert lr[2999] == pytest.approx(BASE_LR)
    assert lr[3000] == pytest.approx(BASE_LR * 0.1)
    assert lr[-1] == pytest.approx(BASE_LR * 0.1)


def test_stage2_returns_finite_audio_of_the_input_shape(backend, long_audio, trace):
    out = _stage2(backend, long_audio, num_iter2=40)

    assert torch.isfinite(torch.as_tensor(out)).all()
    assert out.shape == tuple(long_audio.shape)
