"""White-box CTC adversarial attacks for automatic speech recognition."""

from asr_attacks.attacker import ASRAttacker
from asr_attacks.backends import CTCModuleBackend, HuggingFaceCTCBackend
from asr_attacks.compat import ASRAttacks

__all__ = [
    "ASRAttacker",
    "ASRAttacks",
    "CTCModuleBackend",
    "HuggingFaceCTCBackend",
]

__version__ = "0.2.0"
