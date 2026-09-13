# Contributing

The snapshot of the original research dump lives on `working-main`. New work goes to `main`.

## Setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
ruff check src tests examples
pytest
```

Open a pull request against `main` with a short description of the robustness evaluation change. Do not commit notebook outputs, model weights, or audio files.

## Code notes

- Prefer `ASRAttacker` over the `ASRAttacks` compatibility facade.
- Keep attack math in `src/asr_attacks/attacks/` and model I/O in `src/asr_attacks/backends/`.
- If an implementation differs from the named paper, say so in the docstring (see the CW-style attack).
