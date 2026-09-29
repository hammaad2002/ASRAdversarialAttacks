# Installation

Python 3.10 or newer is required. PyTorch 2.x is a required dependency; install
the CPU or CUDA build that matches your machine **before** this package if you
need a specific wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

## From PyPI

```bash
pip install asr-attacks
```

## From GitHub (until PyPI is published)

```bash
pip install "asr-attacks @ git+https://github.com/hammaad2002/ASRAdversarialAttacks.git"
```

## Extras

| Extra | Installs | Use when |
| --- | --- | --- |
| `wav2vec2` | `torchaudio` | Torchaudio wav2vec2 pipelines |
| `hf` | `transformers` | Hugging Face `AutoModelForCTC` |
| `dev` | pytest, ruff, pre-commit | Contributors |
| `docs` | MkDocs, mkdocstrings | Building this site |
| `build` | build, twine | Cutting a PyPI release |

```bash
pip install "asr-attacks[wav2vec2]"
pip install "asr-attacks[hf]"
pip install -e ".[dev,wav2vec2,docs]"
```

Combine extras with commas: `asr-attacks[wav2vec2,hf]`.

## Editable source checkout

```bash
git clone https://github.com/hammaad2002/ASRAdversarialAttacks.git
cd ASRAdversarialAttacks
python -m pip install -e ".[dev,wav2vec2,docs]"
```

## Verify

```bash
python -c "import asr_attacks; print(asr_attacks.__version__)"
pytest
```
