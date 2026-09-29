# Contributing

## Setup

```bash
python -m pip install -e ".[dev,docs,wav2vec2]"
pre-commit install
```

## Checks

```bash
ruff check src tests examples AdversarialAttacks.py
pytest
mkdocs build --strict
```

Preview docs:

```bash
mkdocs serve
```

Open a pull request against `main`. Do not commit notebook outputs, model
weights, or audio files.

## Code notes

- Prefer `ASRAttacker` when you need a backend (Hugging Face or a custom CTC
  model). `ASRAttacks` is the same attacks with a module-plus-labels constructor.
- Keep attack math in `src/asr_attacks/attacks/` and model I/O in
  `src/asr_attacks/backends/`.
- If an implementation differs from the named paper, say so in the docstring
  and in [algorithms](guide/algorithms.md).
- Public APIs need Google-style docstrings; MkDocs pulls them into the API
  reference.
