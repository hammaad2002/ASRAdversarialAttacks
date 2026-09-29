# Changelog

## Unreleased

### Breaking

- **CW `c`:** now weights the CTC (classifier) term in `||delta||_2^2 + c * CTC`, matching Carlini & Wagner 2018 (audio). Previously `c` weighted the L2 term.
- **CW defaults:** `epsilon ≈ 2000/32768`, `c=1.0`, `learning_rate ≈ 10/32768`, `num_iter=5000`, `decrease_factor_eps=0.8`, `optimizer="adam"`, `early_stop=False`, `search_eps=True`.
- **Imperceptible defaults:** stage iterations/learning rates and `alpha=0.05` follow Qin et al. 2019 (int16 hyperparameters scaled to `[-1, 1]`). Stage 1 uses SGD sign steps; stage 2 uses Adam.
- **Imperceptible `c`:** deprecated and ignored (DeprecationWarning if passed). Stage 2 is `CTC + alpha * l_theta` with no `c` factor.

### Added

- Optional `label=` on FGSM, BIM, PGD, CW, and Imperceptible for untargeted true-label attacks.
- `norm` in `{inf, 2, 1}` for FGSM, BIM, and PGD; PGD `restarts`; Imperceptible `mode` / `rooms` / `robust_delta`.
- Audio CW Sec. III-C (`loss="margin"`), Sec. III-F (`target=""`), and `db_bound` / `db_distortion`.

## 0.2.0 - 2026-09-11

First packaged release of `asr-attacks`.

- Add Apache-2.0 license, NOTICE (IBM ART attribution), pyproject.toml, tests, CI, MkDocs, and PyPI packaging metadata.
- Split the single 1,500-line class into backends, attacks, and metrics.
- Encode targets from the model vocabulary instead of a hardcoded wav2vec2 map.
- Project BIM/PGD onto the L-inf ball around the original waveform.
- Return NumPy arrays via `.detach().cpu().numpy()` so CUDA tensors do not crash.
- Replace homemade positional WER with jiwer (Levenshtein). `wer_compute` no longer returns `1 - WER` for targeted evaluation.
- Use `tqdm.auto` so the library imports outside Jupyter.
- Document that the CW-named attack is CTC + L2, not paper-faithful Carlini–Wagner.

## 0.1.0 - 2023-11-07

Initial research code (unpublished).
