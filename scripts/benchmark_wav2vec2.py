#!/usr/bin/env python
"""Benchmark asr-attacks against a real ASR model and write ``docs/guide/benchmark.md``.

Model:   torchaudio ``WAV2VEC2_ASR_BASE_960H`` (wav2vec2 base, fine-tuned on LibriSpeech 960 h)
Data:    the first utterances of ``hf-internal-testing/librispeech_asr_dummy`` (LibriSpeech clean)
Attacks: FGSM, BIM, PGD (L-inf and L2, untargeted and targeted) and the audio C&W attack

This is a local sanity benchmark, not part of CI: it downloads about 370 MB and wants a GPU
(a few minutes on an RTX 3060, hours on a CPU).

    pip install -e ".[wav2vec2,benchmark]"
    python scripts/benchmark_wav2vec2.py            # full run, rewrites docs/guide/benchmark.md
    python scripts/benchmark_wav2vec2.py --quick    # 2 utterances, ~10x fewer steps, smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

import asr_attacks
from asr_attacks import ASRAttacker, CTCModuleBackend
from asr_attacks.attacks.iterative import bim_num_iter
from asr_attacks.metrics import attack_succeeded, db_distortion, word_error_rate
from asr_attacks.text import display_text

REPO = Path(__file__).resolve().parent.parent
DATASET = "hf-internal-testing/librispeech_asr_dummy"
PARQUET = "clean/validation-00000-of-00001.parquet"
TARGET = "OPEN THE DOOR"
SAMPLE_RATE = 16000
LINF = 0.005  # untargeted L-inf budget on the [-1, 1] waveform
LINF_TARGETED = 0.02


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray
    text: str


@dataclass(frozen=True)
class Attack:
    """One benchmark row. Keyword values may be callables ``f(audio)`` (per-utterance)."""

    name: str
    method: str
    norm: str
    budget: str
    targeted: bool
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    attack: Attack
    success: float
    wer_goal: float
    wer_reference: float
    db_peak: float
    db_rms: float
    seconds: float
    steps: str


def l2_budget(rms: float) -> Callable[[np.ndarray], float]:
    """L2 radius of a signal with constant amplitude ``rms``: the L-inf budget's energy twin."""
    return lambda audio: rms * math.sqrt(len(audio))


def build_attacks(scale: int) -> tuple[list[Attack], list[Attack]]:
    """Return ``(untargeted, targeted)`` rows. ``scale`` divides the step counts (--quick)."""

    def steps(n: int) -> int:
        return max(1, n // scale)

    untargeted = [
        Attack("FGSM", "fgsm", "L∞", f"ε = {LINF / 2}", False, {"epsilon": LINF / 2}),
        Attack("FGSM", "fgsm", "L∞", f"ε = {LINF}", False, {"epsilon": LINF}),
        Attack("FGSM", "fgsm", "L2", f"RMS {LINF}", False, {"epsilon": l2_budget(LINF), "norm": 2}),
        Attack("BIM", "bim", "L∞", f"ε = {LINF}", False, {"epsilon": LINF}),
        Attack("BIM", "bim", "L2", f"RMS {LINF}", False, {"epsilon": l2_budget(LINF), "norm": 2}),
        Attack("PGD", "pgd", "L∞", f"ε = {LINF}", False, {"epsilon": LINF, "num_iter": steps(20)}),
        Attack(
            "PGD",
            "pgd",
            "L2",
            f"RMS {LINF}",
            False,
            {"epsilon": l2_budget(LINF), "norm": 2, "num_iter": steps(20)},
        ),
    ]
    targeted = [
        Attack(
            "FGSM",
            "fgsm",
            "L∞",
            f"ε = {LINF_TARGETED}",
            True,
            {"epsilon": LINF_TARGETED},
        ),
        Attack("BIM", "bim", "L∞", f"ε = {LINF_TARGETED}", True, {"epsilon": LINF_TARGETED}),
        Attack(
            "PGD",
            "pgd",
            "L∞",
            f"ε = {LINF_TARGETED}",
            True,
            {"epsilon": LINF_TARGETED, "num_iter": steps(100), "early_stop": True},
        ),
        Attack(
            "PGD",
            "pgd",
            "L2",
            f"RMS {LINF_TARGETED}",
            True,
            {
                "epsilon": l2_budget(LINF_TARGETED),
                "norm": 2,
                "num_iter": steps(100),
                "early_stop": True,
            },
        ),
        Attack(
            "C&W",
            "cw",
            "L∞ + L2",
            "shrinking bound, ε₀ = 0.061",
            True,
            {"num_iter": steps(500), "search_eps": True},
        ),
    ]
    return untargeted, targeted


def load_utterances(count: int, max_seconds: float) -> list[Utterance]:
    """First ``count`` dummy-LibriSpeech utterances of at most ``max_seconds`` seconds.

    Utterances are kept whole (never cropped) so the reference transcript stays valid.
    """
    import pyarrow.parquet as pq
    import soundfile
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset")
    utterances: list[Utterance] = []
    for row in pq.ParquetFile(path).read().to_pylist():
        audio, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != SAMPLE_RATE or len(audio) / rate > max_seconds:
            continue
        utterances.append(Utterance(np.asarray(audio, dtype=np.float32), row["text"].upper()))
        if len(utterances) == count:
            break
    if len(utterances) < count:
        raise SystemExit(f"only {len(utterances)} utterances are <= {max_seconds} s")
    return utterances


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_attack(
    attacker: ASRAttacker,
    attack: Attack,
    utterances: list[Utterance],
    clean: list[str],
    device: torch.device,
    seed: int,
) -> Outcome:
    hits: list[bool] = []
    goal_wer: list[float] = []
    reference_wer: list[float] = []
    peak_db: list[float] = []
    rms_db: list[float] = []
    seconds: list[float] = []
    for index, (utterance, clean_text) in enumerate(zip(utterances, clean, strict=True)):
        torch.manual_seed(seed + index)  # PGD's random start
        kwargs = {k: (v(utterance.audio) if callable(v) else v) for k, v in attack.kwargs.items()}
        if attack.targeted:
            kwargs.update(target=TARGET, targeted=True)

        sync(device)
        started = time.perf_counter()
        adversarial = getattr(attacker, attack.method)(utterance.audio, **kwargs)
        sync(device)
        seconds.append(time.perf_counter() - started)

        if not np.isfinite(adversarial).all():
            raise RuntimeError(f"{attack.name} returned non-finite audio on utterance {index}")
        text = display_text(attacker.decode(adversarial))
        if attack.targeted:
            hits.append(attack_succeeded(text, TARGET, targeted=True))
            goal_wer.append(word_error_rate(TARGET, text))
        else:
            hits.append(text != clean_text)
            goal_wer.append(word_error_rate(clean_text, text))
        reference_wer.append(word_error_rate(utterance.text, text))
        if not np.array_equal(adversarial, utterance.audio):
            peak_db.append(db_distortion(utterance.audio, adversarial))
            rms_db.append(rms_distortion(utterance.audio, adversarial))

    steps = attack.kwargs.get("num_iter")
    return Outcome(
        attack=attack,
        success=float(np.mean(hits)),
        wer_goal=float(np.mean(goal_wer)),
        wer_reference=float(np.mean(reference_wer)),
        db_peak=float(np.mean(peak_db)) if peak_db else float("nan"),
        db_rms=float(np.mean(rms_db)) if rms_db else float("nan"),
        seconds=float(np.mean(seconds)),
        steps=str(steps)
        if steps is not None
        else f"{bim_num_iter(1.0, 0.1)} (paper rule)"  # alpha = epsilon / 10 for any epsilon
        if attack.method == "bim"
        else "1",
    )


def rms_distortion(original: np.ndarray, adversarial: np.ndarray) -> float:
    """Energy ratio of the perturbation to the signal in dB: ``20·log10(‖δ‖₂ / ‖x‖₂)``."""
    delta = np.linalg.norm(adversarial.astype(np.float64) - original.astype(np.float64))
    return float(20.0 * np.log10(delta / np.linalg.norm(original.astype(np.float64))))


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def table(outcomes: list[Outcome], goal: str) -> str:
    header = (
        f"| Attack | Norm | Budget | Steps | Success | WER vs {goal} | WER vs reference "
        "| Peak (dB) | RMS (dB) | Seconds / utterance |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        f"| {o.attack.name} | {o.attack.norm} | {o.attack.budget} | {o.steps} "
        f"| {o.success:.0%} | {o.wer_goal:.2f} | {o.wer_reference:.2f} "
        f"| {o.db_peak:.1f} | {o.db_rms:.1f} | {o.seconds:.2f} |"
        for o in outcomes
    ]
    return "\n".join([header, *rows])


def render(
    untargeted: list[Outcome],
    targeted: list[Outcome],
    *,
    utterances: list[Utterance],
    clean_wer: float,
    device: torch.device,
    max_seconds: float,
    quick: bool,
) -> str:
    seconds = sum(len(u.audio) for u in utterances) / SAMPLE_RATE
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    quick_note = (
        '\n!!! warning "Quick run"\n    This page was generated with `--quick` (a smoke test); '
        "the numbers are not meaningful. Rerun without `--quick`.\n"
        if quick
        else ""
    )
    data = (
        f"first {len(utterances)} utterances of "
        f"[`{DATASET}`](https://huggingface.co/datasets/{DATASET}) "
        f"that are at most {max_seconds:g} s ({seconds:.0f} s in total)"
    )
    return f"""# Real-model benchmark

A sanity check of every attack against a real ASR model, not a robustness leaderboard.
{quick_note}
| | |
| --- | --- |
| Model | torchaudio `WAV2VEC2_ASR_BASE_960H` (wav2vec2 base, LibriSpeech 960 h) |
| Data | {data} |
| Clean WER vs reference | {clean_wer:.2f} |
| Targeted phrase | `{TARGET}` |
| Package | `asr-attacks {asr_attacks.__version__}` at commit `{git_revision()}` |
| Hardware | {gpu}, torch {torch.__version__} |
| Generated | {dt.date.today().isoformat()} by `scripts/benchmark_wav2vec2.py` |

## Untargeted

The attack moves the transcript away from the model's own clean prediction
(`label=None`, the default). *Success* is the share of utterances whose transcript
changed at all, even by a single letter, so read it together with *WER vs clean*, which
measures how far it moved. *WER vs reference* is against the ground truth (compare with
the clean value above).

{table(untargeted, "clean")}

## Targeted

The attack must make the model transcribe `{TARGET}`. *Success* means the transcript
matched the phrase exactly. *WER vs target* is the distance that remains.

{table(targeted, "target")}

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
- {len(utterances)} utterances is enough to catch a broken attack, not to rank attacks
  precisely. Rerun with `--samples` to tighten the numbers.
- GPU runs are not bit-for-bit reproducible (some CUDA kernels are non-deterministic, and a
  100-step attack amplifies that): repeating this benchmark moved a few success rates by
  a couple of utterances. Read differences of a few points as noise.

## Reproduce

```bash
pip install -e ".[wav2vec2,benchmark]"
python scripts/benchmark_wav2vec2.py --samples {len(utterances)} --max-seconds {max_seconds:g}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=20, help="number of utterances")
    parser.add_argument("--max-seconds", type=float, default=8.0, help="longest utterance to use")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "guide" / "benchmark.md")
    parser.add_argument("--quick", action="store_true", help="2 utterances, 10x fewer steps")
    args = parser.parse_args()

    import torchaudio

    device = torch.device(args.device)
    if device.type == "cpu":
        print("warning: running on the CPU; this takes hours (about 5 s per step)", flush=True)
    samples, scale = (2, 10) if args.quick else (args.samples, 1)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval().to(device)
    backend = CTCModuleBackend(model, labels=list(bundle.get_labels()), device=device)
    attacker = ASRAttacker(backend, verbose=False)

    utterances = load_utterances(samples, args.max_seconds)
    clean = [display_text(attacker.decode(u.audio)) for u in utterances]
    clean_wer = float(
        np.mean([word_error_rate(u.text, c) for u, c in zip(utterances, clean, strict=True)])
    )
    print(f"{len(utterances)} utterances, clean WER vs reference {clean_wer:.3f}", flush=True)

    untargeted_rows, targeted_rows = build_attacks(scale)
    results: dict[bool, list[Outcome]] = {False: [], True: []}
    for attack in [*untargeted_rows, *targeted_rows]:
        outcome = run_attack(attacker, attack, utterances, clean, device, args.seed)
        results[attack.targeted].append(outcome)
        print(
            f"{attack.name:5s} {attack.norm:8s} {attack.budget:28s} success {outcome.success:4.0%}"
            f"  WER(goal) {outcome.wer_goal:.2f}  peak {outcome.db_peak:6.1f} dB"
            f"  rms {outcome.db_rms:6.1f} dB  {outcome.seconds:6.2f} s",
            flush=True,
        )

    markdown = render(
        results[False],
        results[True],
        utterances=utterances,
        clean_wer=clean_wer,
        device=device,
        max_seconds=args.max_seconds,
        quick=args.quick,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
