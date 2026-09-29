# Attacks

All methods return a **CPU NumPy** array with the same layout as the input
waveform, clipped to `[-1, 1]`.

Shared arguments:

| Argument | Meaning |
| --- | --- |
| `audio` | One utterance: tensor or ndarray, `(samples,)` or `(1, samples)` |
| `target` | Transcript for targeted attacks |
| `targeted` | Minimize CTC toward `target`, or maximize away from the clean decode / `label` |
| `label` | Ground-truth transcript for untargeted attacks (default: clean greedy decode) |
| `epsilon` | Perturbation budget in the chosen norm (FGSM: single step) |
| `norm` | `"inf"` (default), `2`, or `1` — FGSM, BIM, and PGD |
| `nested` | `True` if you wrap this call in another tqdm loop |
| `early_stop` | Stop when the greedy transcript succeeds |

Every attack supports targeted and untargeted modes (untargeted: model
prediction or `label=`). Untargeted Imperceptible is a package extension.

Untargeted attacks freeze the reference transcript at the start. They do not
re-decode a moving target each step.

## FGSM

One normalized gradient step of size `epsilon`.

```python
adv = attacker.fgsm(waveform, epsilon=0.01, targeted=False)
adv = attacker.fgsm(waveform, epsilon=0.01, label="THE CAT", targeted=False)
adv = attacker.fgsm(waveform, epsilon=0.01, norm=2, targeted=False)
adv = attacker.fgsm(waveform, target="THE CAT", epsilon=0.01, targeted=True)
```

## BIM

Iterative FGSM. Each iterate is projected onto the L-`norm` ball of radius
`epsilon`, then `[-1, 1]`. Defaults: `alpha = epsilon / 10`; `num_iter` follows
the Kurakin paper rule when left as `None`.

```python
adv = attacker.bim(
    waveform,
    target="THE CAT",
    epsilon=0.03,
    alpha=0.005,
    num_iter=40,
    targeted=True,
    early_stop=True,
    nested=False,
)
adv = attacker.bim(waveform, epsilon=0.03, norm=2, label="THE CAT", nested=False)
```

## PGD

Same loop as BIM with a norm-matched random start. Default
`alpha = 2.5 * epsilon / num_iter`. Pass `restarts=` for multiple random
starts; `random_start=False` for BIM-style init.

```python
adv = attacker.pgd(waveform, epsilon=0.03, num_iter=40, targeted=False)
adv = attacker.pgd(waveform, epsilon=0.03, norm=2, restarts=3, nested=False)
adv = attacker.pgd(waveform, epsilon=0.03, random_start=False, targeted=False)
```

## CW (audio)

Audio Carlini & Wagner ([arXiv:1801.01944](https://arxiv.org/abs/1801.01944)):
`||delta||_2^2 + c * CTC` inside an L-inf box, Adam, optional bound shrinking.
`loss="margin"` is Sec. III-C (targeted); `target=""` is Sec. III-F silence.
`early_stop` and `search_eps` cannot both be true.

```python
adv = attacker.cw(
    waveform,
    target="THE CAT",
    epsilon=0.05,
    c=1.0,
    optimizer="adam",
    num_iter=200,
    targeted=True,
    nested=False,
)
adv = attacker.cw(waveform, label="THE CAT", targeted=False, num_iter=200, nested=False)
```

## Imperceptible

Qin et al. ([arXiv:1903.10346](https://arxiv.org/abs/1903.10346)). Default
`mode="imperceptible"` is offline (Sec. 4). `mode="robust"` /
`"imperceptible_robust"` need a `RoomSimulator` via `rooms=`. Untargeted mode
is a package extension. Defaults are long; start smaller when experimenting.

```python
adv = attacker.imperceptible(
    waveform,
    target="THE CAT",
    num_iter1=200,
    num_iter2=100,
    sample_rate=16000,
    nested=False,
)
# Robust modes need rooms:
# from asr_attacks.rooms import RoomSimulator
# rooms = RoomSimulator.from_rirs([...])  # or RoomSimulator(num_rooms=...)
# adv = attacker.imperceptible(waveform, target="THE CAT", mode="robust", rooms=rooms, ...)
```

See [algorithms](algorithms.md) for paper fidelity and extensions.
