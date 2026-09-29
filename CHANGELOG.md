# Changelog

## 0.3.0 - 2026-09-30

The first release meant for PyPI: every attack now follows its paper, the backends and
metrics were audited, and the project has CI, coverage, a documentation site and a
real-model benchmark.

### Breaking

- **CW `c`:** now weights the CTC (classifier) term in `||delta||_2^2 + c * CTC`, matching Carlini & Wagner 2018 (audio). Previously `c` weighted the L2 term.
- **CW defaults:** `epsilon ≈ 2000/32768`, `c=1.0`, `learning_rate ≈ 10/32768`, `num_iter=5000`, `decrease_factor_eps=0.8`, `optimizer="adam"`, `early_stop=False`, `search_eps=True`.
- **Imperceptible defaults:** stage iterations/learning rates and `alpha=0.05` follow Qin et al. 2019 (int16 hyperparameters scaled to `[-1, 1]`). Stage 1 uses SGD sign steps; stage 2 uses Adam.
- **Imperceptible `c`:** deprecated and ignored (DeprecationWarning if passed). Stage 2 is `CTC + alpha * l_theta` with no `c` factor.
### Added

- Optional `label=` on FGSM, BIM, PGD, CW, and Imperceptible for untargeted true-label attacks.
- `norm` in `{inf, 2, 1}` for FGSM, BIM, and PGD; PGD `restarts`; BIM step count from the paper's rule.
- Imperceptible `mode` (`"imperceptible"`, `"robust"`, `"imperceptible_robust"`), `rooms` and `robust_delta`, with `RoomSimulator` (new `asr_attacks.rooms`) for over-the-air robustness. `RoomSimulator.from_rirs` takes your own impulse responses; generating rooms needs the new `rooms` extra (pyroomacoustics).
- Audio CW Sec. III-C (`loss="margin"`), Sec. III-F (`target=""`), and `db_bound` / `db_distortion`.
- `ASRAttacks(..., verbose=)`, and `verbose` on every stage of every attack (see Fixed).
- Documentation site (MkDocs Material) with a guide per attack, a paper-to-code map, the API reference and an ethics page.
- A real-model benchmark: [`scripts/benchmark_wav2vec2.py`](scripts/benchmark_wav2vec2.py) attacks torchaudio's wav2vec2 on LibriSpeech and writes `docs/guide/benchmark.md`. Install its dependencies with the new `benchmark` extra.

### Fixed

- **`verbose=False` silenced nothing.** The progress bars ignored it; PGD's result was swallowed in a real-model run. Every loop in BIM, PGD, CW, the margin refinement and every Imperceptible stage (including the robust modes) now obeys `verbose`.
- **Targets that cannot fit the model's output** (more tokens, plus a blank between repeats, than logit frames) returned NaN audio without a warning. `ctc_loss` now raises a `ValueError` that says how many frames are needed and how many exist, and uses `zero_infinity=True` so genuinely infinite terms cannot poison the gradient.
- **Hugging Face input normalization.** `HuggingFaceCTCBackend` skipped the processor's `do_normalize` step, so attacks optimized a different signal than the model sees in deployment. `logits()` now applies zero-mean/unit-variance normalization in autograd when the feature extractor asks for it.
- **NaN gradient in the psychoacoustic loss** whenever the perturbation was zero or constant over a frame (for example when stage 2 started from unchanged stage-1 audio): the STFT magnitude used `sqrt`, whose slope is infinite at 0. It now uses the complex `abs`, with a zero subgradient there.
- **`librosa` and `numba` crashed the real imperceptible path** (`Numba needs NumPy 1.26 or less`) and forced tests to mock the masking code. The masking threshold now uses a NumPy STFT that reproduces `librosa.stft(center=False)`, and `librosa` (with `numba`) is no longer a dependency.
- **Imperceptible + robust decoded all M rooms on every step** in its IR stages (800 decodes for 80 steps). It now decodes only at the scheduled check steps, as the paper's schedule needs.

### Changed

- `attacks/cw.py` (892 lines mixing three concerns) is split into `cw.py`, `imperceptible.py` and `_optim.py`. Behaviour is unchanged and `from asr_attacks.attacks.cw import imperceptible` keeps working.
- The package is type-checked with mypy, formatted and linted with Ruff, and the test suite measures branch coverage (about 90 %). Tests now execute code paths that used to be skipped silently, notably the CW margin refinement, and pin the schedules of the Imperceptible attack.
- CI runs pre-commit, Ruff and mypy, the tests on Python 3.10–3.12 (Linux) and 3.12 (macOS, Windows) with coverage uploaded to Codecov, and a wheel install smoke test. Releases go to TestPyPI on demand and to PyPI from a GitHub Release, using trusted publishing.

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
