# Changelog

## 0.2.0 - 2026-09-11

Packaged rewrite on `main`. The previous notebook-era tree is on `working-main`.

- Add Apache-2.0 license, NOTICE (IBM ART attribution), pyproject.toml, tests, and CI.
- Split the single 1,500-line class into backends, attacks, and metrics.
- Encode targets from the model vocabulary instead of a hardcoded wav2vec2 map.
- Project BIM/PGD onto the L-inf ball around the original waveform.
- Return NumPy arrays via `.detach().cpu().numpy()` so CUDA tensors do not crash.
- Replace homemade positional WER with jiwer (Levenshtein). `wer_compute` no longer returns `1 - WER` for targeted evaluation.
- Use `tqdm.auto` so the library imports outside Jupyter.
- Document that the CW-named attack is CTC + L2, not paper-faithful Carlini–Wagner.

## 0.1.0 - 2023-11-07

Original research dump (see branch `working-main`).
