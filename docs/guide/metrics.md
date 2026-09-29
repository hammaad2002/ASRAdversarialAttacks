# Metrics

Word error rate is Levenshtein (via [jiwer](https://github.com/jitsi/jiwer)).

`|` is turned into a space before scoring, so wav2vec2-style
`"THE|CAT"` matches `"THE CAT"`.

```python
mean_wer, counts = attacker.wer(["THE CAT SAT"], [adv])
# counts[i] == (substitutions, insertions, deletions)
```

Lower-level helpers:

```python
from asr_attacks.metrics import word_error_rate, alignment_counts, attack_succeeded

word_error_rate("the cat", "the bat")  # 0.5
alignment_counts("the cat", "the bat")  # (1, 0, 0)
attack_succeeded("THE CAT", "THE CAT", targeted=True)  # True
```

Early stopping uses `attack_succeeded`: targeted success is WER 0; untargeted
success is any transcript change.

`ASRAttacks.wer_compute` always returns WER. Its `targeted` argument is unused.
