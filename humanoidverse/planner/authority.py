"""SUPERSEDED detection guard — user ruling 2026-09-04 (P4 prerequisite 4).

Every eval script that reads or writes the artifacts listed in
CURRENT_EVALUATION_AUTHORITY.json must go through this guard at entry:
a superseded artifact path must not be (re)written, and a superseded field
must not be cited. The overlay file is the single authority pointer; the
canonical dir stays read-only and stale files are never edited in place.

Env: BDX_AUTHORITY_FILE (default: the campaign overlay). Missing overlay ->
warning only (guard is inert outside the campaign checkout).

Usage:
  from humanoidverse.planner.authority import guard_eval_entry
  guard_eval_entry("bdx_planner_v2combo/deployment_eval.json",
                   output_path=OUT_DIR / "deployment_eval.json")
"""
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_AUTHORITY = ("/home/tcl/Desktop/start/DISTILL_CAMPAIGN/"
                     "CURRENT_EVALUATION_AUTHORITY.json")

_cache = {"path": None, "doc": None}


class AuthoritySupersededError(RuntimeError):
    pass


def load(force=False):
    path = Path(os.environ.get("BDX_AUTHORITY_FILE", DEFAULT_AUTHORITY))
    if not force and _cache["path"] == path:
        return _cache["doc"]
    _cache["path"] = path
    _cache["doc"] = json.loads(path.read_text()) if path.exists() else None
    if _cache["doc"] is None:
        print(f"[authority] WARNING: overlay not found at {path} "
              f"(guard inert)", file=sys.stderr)
    return _cache["doc"]


def _file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def superseded_entries():
    doc = load()
    return (doc or {}).get("superseded", {})


def check_path_writable(output_path, artifact_key=None):
    """Raise if output_path IS a superseded artifact. Matching is on the
    two-segment relative name (e.g. bdx_planner_v2combo/deployment_eval.json)
    so a same-named rerun inside a timestamped dir stays allowed."""
    p = Path(output_path)
    rel2 = f"{p.parent.name}/{p.name}"
    for key, entry in superseded_entries().items():
        fname = key.split(":")[0]
        if artifact_key and key != artifact_key:
            continue
        if rel2 == fname or (not artifact_key and p.name == Path(fname).name
                             and p.parent.name == Path(fname).parent.name):
            sha = _file_sha256(p) if p.exists() else "(absent)"
            raise AuthoritySupersededError(
                f"{p} is SUPERSEDED ({entry.get('status')}): "
                f"{entry.get('reason')} — write to a timestamped dir instead "
                f"(overlay sha {entry.get('sha256', '?')[:16]}, current "
                f"{str(sha)[:16]})")


def check_field(field_key):
    """Return the superseded entry for a dotted field key (e.g.
    'canonical.json:capability_envelope.vx.validated') or None."""
    return superseded_entries().get(field_key)


def guard_eval_entry(artifact_key, output_path=None):
    """Eval-script entry guard: prints the authority notice and refuses to
    overwrite superseded artifacts."""
    doc = load()
    entry = superseded_entries().get(artifact_key)
    if entry:
        print(f"[authority] {artifact_key} = {entry['status']} "
              f"(do_not_cite={entry.get('do_not_cite', False)})")
    if output_path is not None:
        check_path_writable(output_path, artifact_key=None)
    if doc:
        auth = list((doc.get("authority_of_record") or {}).keys())
        print(f"[authority] authority_of_record: {', '.join(auth)}")


def _test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ov = Path(td) / "overlay.json"
        ov.write_text(json.dumps({
            "superseded": {
                "bdx_planner_v2combo/deployment_eval.json":
                    {"status": "SUPERSEDED", "reason": "stale subset",
                     "sha256": "deadbeef"}}}))
        os.environ["BDX_AUTHORITY_FILE"] = str(ov)
        load(force=True)
        # same two-segment path (parent dir + name) -> refuse; timestamped
        # dir reruns with the same filename stay allowed
        target = Path(td) / "bdx_planner_v2combo" / "deployment_eval.json"
        target.parent.mkdir()
        target.write_text("{}")
        try:
            check_path_writable(target)
            raise AssertionError("should have refused superseded path")
        except AuthoritySupersededError:
            pass
        # same filename under a different (timestamped) parent -> allowed
        allowed = Path(td) / "20260904T1200Z_p0_baseline" / "deployment_eval.json"
        allowed.parent.mkdir()
        check_path_writable(allowed)
        # field lookup
        assert check_field("bdx_planner_v2combo/deployment_eval.json")
        assert check_field("nope") is None
        # missing overlay -> inert
        os.environ["BDX_AUTHORITY_FILE"] = str(Path(td) / "missing.json")
        load(force=True)
        check_path_writable(target)   # no refusal without overlay
    print("authority: all tests passed")


if __name__ == "__main__":
    _test()
