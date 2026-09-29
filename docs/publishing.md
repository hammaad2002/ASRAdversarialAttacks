# Publishing to PyPI

The distribution name is **`asr-attacks`** (plural). That is distinct from the
existing PyPI project `asr-attack`.

The import name is `asr_attacks`.

## One-time PyPI setup

1. Create a PyPI account and enable [trusted publishing](https://docs.pypi.org/trusted-publishers/).
2. Add a publisher for this GitHub repo:
   - Owner: `hammaad2002`
   - Repository: `ASRAdversarialAttacks`
   - Workflow: `publish.yml`
   - Environment: `pypi` (optional but recommended)
3. In GitHub, create an environment named `pypi` if you referenced one.
4. Enable GitHub Pages (Settings → Pages → Deploy from GitHub Actions) so
   the documentation URL in `pyproject.toml` resolves.

Test uploads go to [TestPyPI](https://test.pypi.org/) with the same flow if
you add a second environment.

## Local dry run

```bash
python -m pip install -e ".[build]"
python -m build
python -m twine check dist/*
```

Inspect `dist/*.tar.gz` and `dist/*.whl`. The sdist must contain `LICENSE`
and `NOTICE`.

## Release checklist

1. Bump `__version__` in `src/asr_attacks/__init__.py` and `version` in
   `pyproject.toml`.
2. Update `CHANGELOG.md`, `docs/changelog.md`, and `CITATION.cff`.
3. `pytest` and `mkdocs build --strict`.
4. Tag `v0.2.0` (or the new version) and push the tag, **or** run the
   **Publish** workflow from the Actions tab.
5. Confirm [https://pypi.org/project/asr-attacks/](https://pypi.org/project/asr-attacks/).

The publish workflow builds the wheel/sdist and uploads with OIDC. It does
not use a long-lived PyPI password.

## Install from the release

```bash
pip install asr-attacks[wav2vec2]
```
