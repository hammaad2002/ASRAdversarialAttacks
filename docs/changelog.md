# Changelog

The file in git is [`CHANGELOG.md`](https://github.com/hammaad2002/ASRAdversarialAttacks/blob/main/CHANGELOG.md).

## 0.2.0 - 2026-09-11

First packaged release of `asr-attacks`.

- Add Apache-2.0 license, NOTICE (IBM ART attribution), pyproject.toml, tests, CI, and this documentation site.
- Split the single 1,500-line class into backends, attacks, and metrics.
- Encode targets from the model vocabulary instead of a hardcoded wav2vec2 map.
- Project BIM/PGD onto the L-inf ball around the original waveform.
- Return NumPy arrays via `.detach().cpu().numpy()` so CUDA tensors do not crash.
- Replace homemade positional WER with jiwer (Levenshtein). `wer_compute` no longer returns `1 - WER` for targeted evaluation.
- Use `tqdm.auto` so the library imports outside Jupyter.
- Document that the CW-named attack is CTC + L2, not paper-faithful Carlini–Wagner.

## 0.1.0 - 2023-11-07

Initial research code (unpublished).
