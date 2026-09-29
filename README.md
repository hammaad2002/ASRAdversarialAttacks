# ASR Adversarial Attacks

[![CI](https://github.com/hammaad2002/ASRAdversarialAttacks/actions/workflows/ci.yml/badge.svg)](https://github.com/hammaad2002/ASRAdversarialAttacks/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks/graph/badge.svg)](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks)
[![PyPI](https://img.shields.io/pypi/v/asr-attacks)](https://pypi.org/project/asr-attacks/)
[![Python](https://img.shields.io/pypi/pyversions/asr-attacks)](https://pypi.org/project/asr-attacks/)
[![License](https://img.shields.io/pypi/l/asr-attacks)](LICENSE)
[![Docs](https://github.com/hammaad2002/ASRAdversarialAttacks/actions/workflows/docs.yml/badge.svg)](https://hammaad2002.github.io/ASRAdversarialAttacks/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This package tests an ASR (speech-to-text) model against well-known white-box adversarial attacks.

**Docs:** [hammaad2002.github.io/ASRAdversarialAttacks](https://hammaad2002.github.io/ASRAdversarialAttacks/)

## Install

```bash
pip install asr-attacks[wav2vec2]
```

The development version:

```bash
pip install "asr-attacks[wav2vec2] @ git+https://github.com/hammaad2002/ASRAdversarialAttacks.git"
```

| Extra | For |
| --- | --- |
| `wav2vec2` | torchaudio wav2vec2 pipelines |
| `hf` | Hugging Face CTC models |
| `rooms` | pyroomacoustics, to generate the rooms of `RoomSimulator` (`mode="robust"`) |
| `benchmark` | `scripts/benchmark_wav2vec2.py` |
| `dev` | pytest, pytest-cov, mypy, ruff, pre-commit |
| `docs` | MkDocs |
| `build` | `python -m build` and twine |

Python 3.10+ and PyTorch 2.x are required. Install a CUDA/CPU torch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) first if you need a specific build.

## Supported attacks

Every attack supports targeted and untargeted modes (untargeted: model prediction
or `label=`). Untargeted Imperceptible is a package extension.

| Attack | Paper | Notes |
| --- | --- | --- |
| FGSM | [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572) | Single-step; `norm` in `{inf, 2, 1}` |
| BIM | [Kurakin et al., 2017](https://arxiv.org/abs/1607.02533) | Iterative; `norm` in `{inf, 2, 1}` |
| PGD | [Madry et al., 2018](https://arxiv.org/abs/1706.06083) | Random start + optional `restarts`; `norm` in `{inf, 2, 1}` |
| CW | [Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944) | Audio C&W Sec. III-B/C/F (`\|\|δ\|\|_2² + c·CTC`) |
| Imperceptible | [Qin et al., 2019](https://arxiv.org/abs/1903.10346) | Offline by default; `mode="robust"` needs a `RoomSimulator` |

How they behave on a real model (wav2vec2 on LibriSpeech, success rate, distortion and
run time) is in the [benchmark](https://hammaad2002.github.io/ASRAdversarialAttacks/guide/benchmark/).

## Quickstart

```python
import torch
import torchaudio
from asr_attacks import ASRAttacker, CTCModuleBackend

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

backend = CTCModuleBackend(model, labels=list(bundle.get_labels()), device=device)
attacker = ASRAttacker(backend)

waveform, sample_rate = torchaudio.load("example.wav")
assert sample_rate == 16000

print("clean:", attacker.decode(waveform))
adv = attacker.fgsm(waveform, epsilon=0.01, targeted=False)
adv = attacker.fgsm(waveform, epsilon=0.01, label="THE CAT", norm=2, targeted=False)
print("adversarial:", attacker.decode(adv))
```

Targeted BIM:

```python
adv = attacker.bim(
    waveform,
    target="THE CAT SAT",
    epsilon=0.03,
    alpha=0.005,
    num_iter=50,
    targeted=True,
    early_stop=True,
    nested=False,
)
mean_wer, counts = attacker.wer(["THE CAT SAT"], [adv])
```


Hugging Face CTC:

```python
from asr_attacks import ASRAttacker, HuggingFaceCTCBackend

backend = HuggingFaceCTCBackend("facebook/wav2vec2-base-960h", device="cpu")
attacker = ASRAttacker(backend)
```

Alternatively, wrap a CTC module with `ASRAttacks`:

```python
from asr_attacks import ASRAttacks

attacks = ASRAttacks(model, "cpu", list(bundle.get_labels()))
adv = attacks.FGSM_ATTACK(waveform, epsilon=0.01, targeted=False)
```

## Coverage

Tests run on Python 3.10–3.12 (Linux) and 3.12 (macOS, Windows) for every pull request,
with branch coverage reported to [Codecov](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks).

[![Coverage sunburst](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks/graphs/sunburst.svg)](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks)

## Responsible use

These methods exist to measure and defend ASR systems. Use them only on models and data you are authorized to evaluate. Do not use this project to interfere with production speech systems you do not own.

## Cite

See [`CITATION.cff`](CITATION.cff). If you use the imperceptible attack, also cite Qin et al. (2019) and IBM ART.

## License

Apache License 2.0. Psychoacoustic helpers are derived from the IBM Adversarial Robustness Toolbox; see [`NOTICE`](NOTICE).
