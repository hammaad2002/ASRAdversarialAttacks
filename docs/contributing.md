# Contributing

## Setup

```bash
python -m pip install -e ".[dev,docs,wav2vec2,hf,rooms]"
pre-commit install
```

`pre-commit install` runs Ruff (lint and format), notebook-output stripping, and basic
file hygiene checks before every commit. Install a CPU-only PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) if you do not
need a GPU build.

## Checks

CI runs all of these on every pull request:

```bash
pre-commit run --all-files
mypy
pytest --cov=asr_attacks
mkdocs build --strict
```

Coverage is uploaded to [Codecov](https://codecov.io/gh/hammaad2002/ASRAdversarialAttacks);
a pull request should not lower project coverage by more than 1 %, and new lines should
be at least 70 % covered. Prefer tests that fail when the code is wrong: assert on
values, and make sure the branch you are testing actually runs (`--cov-report=term-missing`
shows what was skipped).

Preview docs:

```bash
mkdocs serve
```

Open a pull request against `main`. Do not commit notebook outputs, model weights, or
audio files. Keep commits logical (one feature or fix each): pull requests are rebased,
not squashed.

## Code notes

- Prefer `ASRAttacker` when you need a backend (Hugging Face or a custom CTC
  model). `ASRAttacks` is the same attacks with a module-plus-labels constructor.
- Keep attack math in `src/asr_attacks/attacks/` and model I/O in
  `src/asr_attacks/backends/`.
- If an implementation differs from the named paper, say so in the docstring
  and in [algorithms](guide/algorithms.md).
- Public APIs need Google-style docstrings; MkDocs pulls them into the API
  reference.
