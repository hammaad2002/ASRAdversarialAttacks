# Real-model benchmark

A sanity check of every attack against a real ASR model, not a robustness leaderboard.

| | |
| --- | --- |
| Model | torchaudio `WAV2VEC2_ASR_BASE_960H` (wav2vec2 base, LibriSpeech 960 h) |
| Data | first 20 utterances of [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) that are at most 8 s (87 s in total) |
| Clean WER vs reference | 0.03 |
| Targeted phrase | `OPEN THE DOOR` |
| Package | `asr-attacks 0.3.0` at commit `226082e` |
| Hardware | NVIDIA GeForce RTX 3060, torch 2.11.0+cu128 |
| Generated | 2026-09-30 by `scripts/benchmark_wav2vec2.py` |

## Untargeted

The attack moves the transcript away from the model's own clean prediction
(`label=None`, the default). *Success* is the share of utterances whose transcript
changed at all, even by a single letter, so read it together with *WER vs clean*, which
measures how far it moved. *WER vs reference* is against the ground truth (compare with
the clean value above).

| Attack | Norm | Budget | Steps | Success | WER vs clean | WER vs reference | Peak (dB) | RMS (dB) | Seconds / utterance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FGSM | L∞ | ε = 0.0025 | 1 | 80% | 0.14 | 0.14 | -45.4 | -29.0 | 0.07 |
| FGSM | L∞ | ε = 0.005 | 1 | 80% | 0.16 | 0.16 | -39.4 | -23.0 | 0.06 |
| FGSM | L2 | RMS 0.005 | 1 | 75% | 0.13 | 0.14 | -11.7 | -23.0 | 0.07 |
| BIM | L∞ | ε = 0.005 | 13 (paper rule) | 100% | 0.54 | 0.54 | -39.4 | -31.1 | 0.52 |
| BIM | L2 | RMS 0.005 | 13 (paper rule) | 100% | 0.43 | 0.43 | -26.6 | -31.6 | 0.59 |
| PGD | L∞ | ε = 0.005 | 20 | 100% | 0.52 | 0.53 | -39.4 | -27.0 | 0.87 |
| PGD | L2 | RMS 0.005 | 20 | 100% | 0.59 | 0.59 | -23.5 | -23.0 | 0.88 |

## Targeted

The attack must make the model transcribe `OPEN THE DOOR`. *Success* means the transcript
matched the phrase exactly. *WER vs target* is the distance that remains.

| Attack | Norm | Budget | Steps | Success | WER vs target | WER vs reference | Peak (dB) | RMS (dB) | Seconds / utterance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FGSM | L∞ | ε = 0.02 | 1 | 0% | 3.32 | 0.36 | -27.4 | -10.9 | 0.04 |
| BIM | L∞ | ε = 0.02 | 13 (paper rule) | 0% | 1.45 | 0.91 | -27.4 | -19.8 | 0.57 |
| PGD | L∞ | ε = 0.02 | 100 | 40% | 0.37 | 0.98 | -27.4 | -15.5 | 5.59 |
| PGD | L2 | RMS 0.02 | 100 | 85% | 0.10 | 0.96 | -14.2 | -10.9 | 4.32 |
| C&W | L∞ + L2 | shrinking bound, ε₀ = 0.061 | 500 | 85% | 0.05 | 0.96 | -42.7 | -32.8 | 22.36 |

## How to read this

- **Peak (dB)** is Carlini & Wagner's relative loudness of the perturbation,
  `20·log10(max|δ|) − 20·log10(max|x|)`. **RMS (dB)** is the energy ratio
  `20·log10(‖δ‖₂ / ‖x‖₂)`. More negative is quieter for both (−20 dB is a tenth of the
  reference amplitude, −40 dB a hundredth). Both are averaged over the utterances whose
  audio the attack changed.
- **L2 budgets** are the L2 norm of a constant-amplitude signal (`RMS · √samples`), so an
  L2 row is allowed the same energy as the L∞ row beside it. Compare norms with the **RMS**
  column: an L2 attack may put its energy on a few loud samples, which the peak measure
  penalises even when the total energy is the same.
- **Step counts differ a lot.** FGSM is a single gradient. BIM follows the paper's rule,
  13 steps of `ε/10`, which is enough to change a transcript but rarely enough to reach a
  specific phrase, so it trails PGD (100 steps, random start, early stop) in the targeted
  table. C&W runs 500 steps; its time per utterance is correspondingly higher.
- **C&W** starts from a loose bound and shrinks it after every success, so its distortion is
  the smallest bound it found, not the starting one. It is stopped after a fixed step count
  rather than at convergence: more steps would typically raise its success rate or lower
  its distortion further.
- Success is measured on the digital waveform. These attacks are not claimed to survive
  playback over the air; use `mode="robust"` for that (see the
  [Imperceptible guide](attacks.md#imperceptible)).
- 20 utterances is enough to catch a broken attack, not to rank attacks
  precisely. Rerun with `--samples` to tighten the numbers.
- GPU runs are not bit-for-bit reproducible (some CUDA kernels are non-deterministic, and a
  100-step attack amplifies that): repeating this benchmark moved a few success rates by
  a couple of utterances. Read differences of a few points as noise.

## Reproduce

```bash
pip install -e ".[wav2vec2,benchmark]"
python scripts/benchmark_wav2vec2.py --samples 20 --max-seconds 8
```
