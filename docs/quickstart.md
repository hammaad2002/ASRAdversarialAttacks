# Quickstart

This example uses torchaudio wav2vec2. Install `asr-attacks[wav2vec2]` first.

```python
import torch
import torchaudio
from asr_attacks import ASRAttacker, CTCModuleBackend

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

backend = CTCModuleBackend(model, labels=list(bundle.get_labels()), device=device)
attacker = ASRAttacker(backend)

waveform, sample_rate = torchaudio.load("example.wav")
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

print("clean:", attacker.decode(waveform))
adv = attacker.fgsm(waveform, epsilon=0.01, targeted=False)
print("adversarial:", attacker.decode(adv))
```

## Targeted BIM

Targets may be a string (`"THE CAT SAT"`), a wav2vec2 character list
(`list("THE|CAT|SAT")`), or a one-element list. Spaces become `|` when that
symbol is in the model vocabulary.

```python
target = "THE CAT SAT"
adv = attacker.bim(
    waveform,
    target=target,
    epsilon=0.03,
    alpha=0.005,
    num_iter=50,
    targeted=True,
    early_stop=True,
    nested=False,
)
mean_wer, counts = attacker.wer([target], [adv])
print(mean_wer, counts)
```

## Hugging Face CTC

```python
from asr_attacks import ASRAttacker, HuggingFaceCTCBackend

backend = HuggingFaceCTCBackend("facebook/wav2vec2-base-960h", device="cpu")
attacker = ASRAttacker(backend)
```

Requires `pip install asr-attacks[hf]`.

## Tiny model without weights

`examples/quickstart.py` runs FGSM on a one-layer convolutional CTC stand-in so
you can exercise the API without downloading wav2vec2.
