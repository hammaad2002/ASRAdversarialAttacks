# Responsible use

`asr-attacks` is a robustness-evaluation library. Use it only on models and
recordings you are authorized to test: your own checkpoints, licensed
benchmarks, or systems whose owners have asked for a white-box evaluation.

Do not use it to interfere with production speech systems, voice
authentication, or other people's devices.

Attacks here are **white-box**: they need differentiable CTC logits. They are
not a recipe for attacking a closed API.

If you publish numbers, cite this package ([`CITATION.cff`](https://github.com/hammaad2002/ASRAdversarialAttacks/blob/main/CITATION.cff)),
the attack papers you actually implemented, and IBM ART when you use the
imperceptible stage. Read [algorithms](algorithms.md) for how the audio CW attack maps to
Carlini & Wagner (2018), not the 2017 image C&W paper.

Security issues in the library itself: see
[`SECURITY.md`](https://github.com/hammaad2002/ASRAdversarialAttacks/blob/main/SECURITY.md).
