# ASR Adversarial Attacks

White-box CTC attacks for evaluating the robustness of automatic speech recognition models you own or have permission to test.

The original 2023 notebook dump is preserved on the [`working-main`](https://github.com/hammaad2002/ASRAdversarialAttacks/tree/working-main) branch. This `main` branch is the packaged library.

## Supported attacks

| Attack | Paper | Notes |
| --- | --- | --- |
| FGSM | [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572) | Single-step L-inf sign gradient |
| BIM | [Kurakin et al., 2017](https://arxiv.org/abs/1607.02533) | Iterative FGSM with L-inf projection |
| PGD | [Madry et al., 2018](https://arxiv.org/abs/1706.06083) | BIM with random start inside the L-inf ball |
| CW-style | [Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944) | CTC + L2 inside an L-inf box. **Not** paper-faithful C&W (no tanh change of variables, no binary search on `c`) |
| Imperceptible | [Qin et al., 2019](https://arxiv.org/abs/1903.10346) | Two-stage attack; psychoacoustic stage follows IBM ART |

Backends: torchaudio wav2vec2 (default) and optional Hugging Face CTC models via `pip install asr-attacks[hf]`.

## Install

```bash
git clone https://github.com/hammaad2002/ASRAdversarialAttacks.git
cd ASRAdversarialAttacks
python -m pip install -e ".[dev]"
```

Python 3.10+ is required. For torchaudio wav2vec2 or Hugging Face models:

```bash
python -m pip install -e ".[dev,wav2vec2]"
python -m pip install -e ".[dev,hf]"
```

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

The old `ASRAttacks.FGSM_ATTACK` names still work:

```python
from asr_attacks import ASRAttacks
attacks = ASRAttacks(model, "cpu", list(bundle.get_labels()))
adv = attacks.FGSM_ATTACK(waveform, epsilon=0.01, targeted=False)
```

## Responsible use

These methods exist to measure and defend ASR systems. Use them only on models and data you are authorized to evaluate. Do not use this project to interfere with production speech systems you do not own.

## Cite

See [`CITATION.cff`](CITATION.cff). If you use the imperceptible attack, also cite Qin et al. (2019) and IBM ART.

## License

Apache License 2.0. Psychoacoustic helpers are derived from the IBM Adversarial Robustness Toolbox; see [`NOTICE`](NOTICE).
