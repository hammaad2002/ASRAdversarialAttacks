# Publishing to PyPI

The distribution name is **`asr-attacks`** (plural). That is distinct from the
existing PyPI project `asr-attack`.

The import name is `asr_attacks`.

## One-time setup

**PyPI and TestPyPI.** Both use [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored anywhere. Until the project exists, add a *pending* publisher
(Account → Publishing) on each index:

| Index | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- |
| [pypi.org](https://pypi.org/manage/account/publishing/) | `hammaad2002` | `ASRAdversarialAttacks` | `publish.yml` | `pypi` |
| [test.pypi.org](https://test.pypi.org/manage/account/publishing/) | `hammaad2002` | `ASRAdversarialAttacks` | `publish.yml` | `testpypi` |

**GitHub** (Settings):

1. *Environments* → create `pypi` and `testpypi`. Adding yourself as a required reviewer
   on `pypi` gives a last chance to stop an upload.
2. *Pages* → Source: **GitHub Actions**, so the documentation URL in `pyproject.toml`
   resolves.
3. *Secrets and variables → Actions* → add `CODECOV_TOKEN` (from
   [codecov.io](https://about.codecov.io/), after adding the repository there). Without it
   the coverage upload is skipped and CI still passes.
4. *Branches* → protect `main`: require a pull request and the **CI passed** status check,
   and allow **Rebase and merge** so each feature commit stays in the history.

## Local dry run

```bash
python -m pip install -e ".[build]"
python -m build
python -m twine check --strict dist/*
```

Inspect `dist/*.tar.gz` and `dist/*.whl`. The sdist must contain `LICENSE` and `NOTICE`.

## Release checklist

1. Branch `release/X.Y.Z`. Bump `__version__` in `src/asr_attacks/__init__.py` and
   `version` in `pyproject.toml`; update `CHANGELOG.md`, `docs/changelog.md` and
   `CITATION.cff` (version and `date-released`).
2. Run everything CI runs:

   ```bash
   pre-commit run --all-files
   mypy
   pytest --cov=asr_attacks
   mkdocs build --strict
   python -m build && python -m twine check --strict dist/*
   ```

3. Open a pull request. When **CI passed** is green, **Rebase and merge**.
4. Rehearse on TestPyPI: *Actions → Publish → Run workflow* on `main`. Then, in a fresh
   virtual environment:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ "asr-attacks==X.Y.Z"
   python -c "import asr_attacks; print(asr_attacks.__version__)"
   ```

5. Tag `vX.Y.Z` on `main` and publish a GitHub Release from the tag. The **Publish**
   workflow builds the artifacts once and uploads them to PyPI. It refuses to run when the
   tag is not `v` plus the version in `pyproject.toml`.
6. Confirm [pypi.org/project/asr-attacks](https://pypi.org/project/asr-attacks/), the
   documentation site, and the README badges.

Uploads use OIDC; there is no long-lived PyPI password. A manual run of the workflow can
only reach TestPyPI, never PyPI.

## Install from the release

```bash
pip install asr-attacks[wav2vec2]
```
