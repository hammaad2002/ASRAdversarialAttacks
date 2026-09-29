# Backends

Every attack talks to the model through [`ASRBackend`](../api/backends.md):
`logits`, `encode`, and `decode`. You do not pass a raw `nn.Module` into
`ASRAttacker`.

## CTCModuleBackend

Wraps a PyTorch CTC module such as torchaudio wav2vec2. `labels` must be the
vocabulary **in index order** (`bundle.get_labels()`). Encoding uses that
vocabulary; it is not hardcoded to wav2vec2.

```python
from asr_attacks import CTCModuleBackend

backend = CTCModuleBackend(
    model,
    labels=list(bundle.get_labels()),
    device="cuda",
    freeze_model=True,
)
```

The backend puts the model in eval mode and, by default, freezes parameters so
gradients flow through the waveform only.

Input audio is **one** utterance: `(samples,)` or `(1, samples)`, float32,
typically in `[-1, 1]` at 16 kHz for wav2vec2. The model returns CTC logits
`(1, time, vocab)`, not text.

## HuggingFaceCTCBackend

Wraps `AutoModelForCTC`. Pass a hub id or an already-loaded model plus
processor and tokenizer.

```python
from asr_attacks import HuggingFaceCTCBackend

backend = HuggingFaceCTCBackend("facebook/wav2vec2-base-960h", device="cuda")
```

`encode` uses `tokenizer.encode(..., add_special_tokens=False)`. `decode` uses
`processor.batch_decode` on greedy ids. The CTC blank id is
`model.config.pad_token_id` when set.

### Input normalization

Most wav2vec2-family checkpoints expect each utterance to be normalized to zero
mean and unit variance. When the processor's feature extractor has
`do_normalize=True`, the backend applies exactly that transform inside `logits`,
`(x - mean) / sqrt(var + 1e-7)`, **in autograd**. Attacks therefore perturb the
raw waveform you pass in (and `epsilon` is measured on it), while the loss and
gradients are those of the normalized audio the model actually receives.
Checkpoints with `do_normalize=False` get the waveform untouched.

`CTCModuleBackend` passes the waveform to your module as-is. If your
checkpoint needs normalized input, do it inside the module's `forward` so that
it stays differentiable.

## Custom models

Subclass `ASRBackend` and implement the three methods. `logits` must be
differentiable with respect to the waveform and return shape
`(batch, time, vocab)`.
