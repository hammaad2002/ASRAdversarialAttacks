# asr-attacks

This package tests an ASR (speech-to-text) model against well-known white-box
adversarial attacks.

ASR turns a waveform into text. White-box here means you can run the model in
PyTorch and take gradients through it. An attack adds a small perturbation to
the audio so the decoded transcript changes, or becomes a phrase you choose.

## How a call works

The model is a normal CTC network: **audio in, logits out** — not a string.
`asr-attacks` takes those logits, runs CTC loss, and steps the waveform.

```python
from asr_attacks import ASRAttacker, CTCModuleBackend

# `model` is an nn.Module: waveform -> CTC logits of shape (1, time, vocab)
# `labels[i]` is the token for logits[..., i] (e.g. list(bundle.get_labels()))
backend = CTCModuleBackend(model, labels=list(bundle.get_labels()), device=device)
attacker = ASRAttacker(backend)

# One mono clip: (samples,) or (1, samples), float32, typically in [-1, 1]
print(attacker.decode(waveform))  # greedy CTC -> str, e.g. "THE|CAT"

# Untargeted FGSM: one signed-gradient step of size epsilon
adv = attacker.fgsm(waveform, epsilon=0.01, targeted=False)
# `adv` is a NumPy array, same layout as the prepared input, clipped to [-1, 1]
print(attacker.decode(adv))
```

| Piece | What it is |
| --- | --- |
| `waveform` | One 1-D clip. `(samples,)` or `(1, samples)` — the layout `torchaudio.load` gives you for mono |
| `model(...)` | CTC **logits** `(1, time, vocab)`. `time` is acoustic frames, not characters in the transcript |
| `attacker.decode(...)` | Argmax over `vocab`, collapse repeats, drop the CTC blank, join `labels` into a string |
| `attacker.fgsm(...)` | Perturb the waveform. `targeted=False` pushes the transcript away from the clean decode |

Pass **one utterance per call**. `(2, samples)` is two files stacked, not “stereo”
or a supported batch — attacks error, and decode would only look at the first
row. `(1, 1, samples)` is also rejected; `squeeze` extra channel axes first.

For a full script, Hugging Face models, and targeted attacks, see
[Quickstart](quickstart.md).

## Attacks

| Method | Paper | What this library implements |
| --- | --- | --- |
| FGSM | [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572) | Single-step L-inf sign gradient |
| BIM | [Kurakin et al., 2017](https://arxiv.org/abs/1607.02533) | Iterative FGSM with L-inf projection |
| PGD | [Madry et al., 2018](https://arxiv.org/abs/1706.06083) | BIM with a random start inside the L-inf ball |
| CW | [Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944) | Audio C&W Sec. III-B/C/F (`\|\|δ\|\|_2² + c·CTC`) |
| Imperceptible | [Qin et al., 2019](https://arxiv.org/abs/1903.10346) | Two-stage attack; psychoacoustic stage follows IBM ART |

## Install

```bash
pip install asr-attacks[wav2vec2]
```

Until the package is on PyPI, install from GitHub:

```bash
pip install "asr-attacks[wav2vec2] @ git+https://github.com/hammaad2002/ASRAdversarialAttacks.git"
```

See [installation](install.md) for extras, Python versions, and a source checkout.

## Next

- [Quickstart](quickstart.md)
- [Attack parameters](guide/attacks.md)
- [API reference](api/attacker.md)
