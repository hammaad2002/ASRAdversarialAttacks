"""Keep the attack modules small and the old import paths working."""

from __future__ import annotations

import sys
from pathlib import Path

import asr_attacks

MAX_MODULE_LINES = 1000


def test_no_source_module_exceeds_the_line_budget():
    root = Path(asr_attacks.__file__).parent
    too_long = {
        str(path.relative_to(root)): count
        for path in root.rglob("*.py")
        if (count := len(path.read_text().splitlines())) > MAX_MODULE_LINES
    }
    assert not too_long, f"split these modules (limit {MAX_MODULE_LINES} lines): {too_long}"


def test_imperceptible_is_still_importable_from_cw():
    from asr_attacks.attacks import cw as cw_function
    from asr_attacks.attacks import imperceptible as public_imperceptible
    from asr_attacks.attacks.cw import cw, imperceptible
    from asr_attacks.attacks.imperceptible import imperceptible as home

    assert imperceptible is home is public_imperceptible
    assert cw is cw_function


def test_cw_module_declares_its_public_names():
    module = sys.modules["asr_attacks.attacks.cw"]
    assert set(module.__all__) == {"cw", "imperceptible"}


def test_top_level_package_exports_every_attack():
    for name in ("ASRAttacker", "ASRAttacks", "CTCModuleBackend", "ASRBackend"):
        assert hasattr(asr_attacks, name)
    attacker = asr_attacks.ASRAttacker
    for method in ("fgsm", "bim", "pgd", "cw", "imperceptible"):
        assert callable(getattr(attacker, method))
