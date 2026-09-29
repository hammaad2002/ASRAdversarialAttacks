# Algorithms

This page states where implementations follow the named papers and where they
are package extensions. Use it when citing results.

## Labels (shared)

Every attack supports **targeted** and **untargeted** mode. Untargeted attacks
maximize CTC loss against a fixed reference: the model's clean greedy decode
by default (ART-style), or a ground-truth transcript via `label=` (Goodfellow
2015 / Kurakin 2017 / Madry 2018 true-label setup). Passing `label=` with
`targeted=True` raises. Untargeted Imperceptible is a **package extension**
beyond Qin et al.

**Label leaking:** one-step true-label FGSM can leak the label when used for
adversarial training ([Kurakin et al., 2017](https://arxiv.org/abs/1611.01236)).
Iterative attacks barely leak.

## FGSM

L-inf sign-of-gradient step on CTC loss
([Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)). Targeted attacks
negate the sign so CTC is minimized toward the encoded target ([Kurakin et
al., 2017](https://arxiv.org/abs/1611.01236)).

`norm=2` / `norm=1` are the Fast Gradient Method (Kurakin et al., 2017), with
L1 normalized as in ART (`g / ||g||_1`).

## BIM

Kurakin-style iterated FGSM
([Kurakin et al., 2017](https://arxiv.org/abs/1607.02533)) with projection onto
the L-`norm` ball around the **original** waveform, then clip to `[-1, 1]`.
Default `alpha = epsilon / 10`; default `num_iter` uses the paper rule
`ceil(min(eps/alpha + 4, 1.25 * eps/alpha))`.

`norm=2` / `norm=1` ("L2 / L1 BIM") are package extensions; the paper is L-inf
only. Early stopping is also an extension. Least-likely-class BIM is covered
by `targeted=True` with a user-chosen transcript (ASR has no natural
least-likely class).

## PGD

Madry-style PGD ([Madry et al., 2018](https://arxiv.org/abs/1706.06083)): same
inner loop as BIM with a norm-matched random start in the L-`norm` ball.
Default `alpha = 2.5 * epsilon / num_iter`.

- **Restarts:** `restarts > 1` runs independent random starts and keeps the
  strongest result (success first, then best CTC loss). Requires
  `random_start=True`.
- **Norms:** L-inf (default); L2 cited to Sec. 5 of the paper; L1 follows ART
  and is a package extension. Early stopping is an extension.
- PGD with the CW margin loss is not provided (no direct CTC equivalent).

## CW (audio)

Paper-faithful audio Carlini & Wagner
([Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944)), **not** the 2017
image C&W paper (no tanh change of variables, no binary search on `c`).

| Section | Behavior |
| --- | --- |
| III-B | Minimize `\|\|delta\|\|_2^2 + c * CTC` inside an L-inf box (`epsilon` or `db_bound`), Adam, bound shrinks by `decrease_factor_eps` after successes |
| III-C | `loss="margin"` (targeted only): refine CTC with a per-frame hinge on logits under a fixed greedy alignment; per-frame `c_i` doubling is a package choice |
| III-F | `target=""`: silence-token hinge; success is an empty displayed transcript |

Untargeted mode (flip CTC sign) is a package extension. Distortion can be
reported with `db_distortion`. Not included: EOT/MP3 robustness (Sec. IV-B).

## Imperceptible ASR

[Qin et al., 2019](https://arxiv.org/abs/1903.10346). Default
`mode="imperceptible"` is the offline Sec. 4 attack:

- Stage 1: CTC-only L-inf bound search with signed gradient steps (Algorithm 1).
- Stage 2: `CTC + alpha * l_theta` with Adam; waveform clamped to `[-1, 1]`.
  CTC stands in for the paper's Lingvo cross-entropy.

`mode="robust"` / `"imperceptible_robust"` need a `RoomSimulator` via
`rooms=` (Sec. 5 / 6, App. D). The masking threshold and PSD transform follow
IBM ART's PyTorch Imperceptible ASR code (Apache-2.0; see `NOTICE`).
Untargeted mode is a package extension. `c` is deprecated and ignored.

## Decoding and encoding

Greedy CTC: argmax, collapse repeats, drop blank. Targets are encoded with
the **backend vocabulary**, not a hardcoded wav2vec2 map.
