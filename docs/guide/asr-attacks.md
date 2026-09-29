# ASRAttacks

`ASRAttacks` wraps a CTC `nn.Module` plus its vocabulary. Use
[`ASRAttacker`](../api/attacker.md) when you already have a backend
(Hugging Face, a custom model, and so on).

```python
from asr_attacks import ASRAttacks

attacks = ASRAttacks(model, device, list(bundle.get_labels()))
adv = attacks.FGSM_ATTACK(waveform, epsilon=0.01, targeted=False)
text = attacks.INFER(waveform)
wer, details = attacks.wer_compute(["THE CAT"], [adv])
```

| `ASRAttacks` | `ASRAttacker` |
| --- | --- |
| `FGSM_ATTACK` | `fgsm` |
| `BIM_ATTACK` | `bim` |
| `PGD_ATTACK` | `pgd` |
| `CW_ATTACK` | `cw` |
| `IMPERCEPTIBLE_ATTACK` | `imperceptible` |
| `INFER` | `decode` |
| `wer_compute` | `wer` |

`PGD_ATTACK` uses a random start inside the L-inf ball by default
(`random_start=True`).
