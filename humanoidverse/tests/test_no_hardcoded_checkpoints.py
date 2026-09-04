"""Lint: production scripts must not hardcode checkpoint filenames.

Allowed: canonical.json, docs/legacy/*, this file, registry.py itself.
Rationale: audit 2026-08-24 — silent legacy loading + preprocess skew both
entered through hardcoded defaults.
"""
import re
from pathlib import Path

REPO = Path(__file__).parents[2] / "humanoidverse"
ALLOWED_FILES = {"canonical.json"}
ALLOWED_DIRS = {"docs"}
PATTERN = re.compile(r"p\d?_student[\w.]*\.pt|p1_ae[\w.]*\.pt|legacy_audit[\w.]*\.pt")

def test_no_hardcoded_checkpoints():
    hits = []
    for p in (REPO / "scripts").glob("*.py"):
        if "audit" in p.name or "train_" in p.name:   # trainers write tags; auditors inspect
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()[:70]}")
    assert not hits, "hardcoded checkpoint references:\n" + "\n".join(hits)

def test_preprocess_fail_closed_default():
    import humanoidverse.planner.dataset as D
    assert D._active_preprocess is None, "module default must stay None (fail-closed)"

if __name__ == "__main__":
    test_no_hardcoded_checkpoints()
    test_preprocess_fail_closed_default()
    print("lint passed")
