"""Auto-generated closure status — kills hand-counted drift (GPT directive #4).
Formal reports must quote this file, never hand-typed numbers."""
import json, re, subprocess, datetime
from pathlib import Path

REPO = Path('/home/tcl/Desktop/start/BFM-zero')
SNAP = Path('/home/tcl/Desktop/start/bfmzero-archive/bfm-planner-release-rc1')

def count_registry_consumers():
    pat = re.compile(r'resolve_canonical')
    hits = []
    for p in (REPO / 'humanoidverse' / 'scripts').glob('*.py'):
        if pat.search(p.read_text()):
            hits.append(p.name)
    return sorted(hits)

def count_hardcoded():
    pat = re.compile(r'p\d?_student[\w.]*\.pt|p1_ae[\w.]*\.pt|legacy_audit[\w.]*\.pt')
    hits = []
    for p in (REPO / 'humanoidverse' / 'scripts').glob('*.py'):
        if 'audit' in p.name or 'train_' in p.name:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append(f'{p.name}:{i}')
    return hits

def count_tests():
    r = subprocess.run(['.venv/bin/python', 'humanoidverse/tests/test_planner_contracts.py'],
                       capture_output=True, text=True, cwd=REPO)
    m = re.search(r'(\d+) contract tests passed', r.stdout)
    contracts = int(m.group(1)) if m else 0
    r2 = subprocess.run(['.venv/bin/python', 'humanoidverse/tests/test_contact_detection.py'],
                        capture_output=True, text=True, cwd=REPO)
    contact = 'passed' in r2.stdout
    r3 = subprocess.run(['.venv/bin/python', 'humanoidverse/tests/test_no_hardcoded_checkpoints.py'],
                        capture_output=True, text=True, cwd=REPO)
    lint = 'passed' in r3.stdout
    return {'contract_tests': contracts, 'contact_regression': contact,
            'checkpoint_lint': lint}

lineage = json.loads((REPO / 'humanoidverse/data/bdx_planner_v2combo/lineage_audit_result.json').read_text())
snap_files = int(subprocess.run(['git', 'ls-files'], capture_output=True, text=True,
                                cwd=SNAP).stdout.strip().count('\n') + 1)
reg = json.loads((REPO / 'humanoidverse/data/bdx_planner_v2combo/canonical.json').read_text())
dep = json.loads((REPO / 'humanoidverse/data/bdx_planner_v2combo/deployment_eval.json').read_text())

status = {
    'generated': datetime.datetime.now().isoformat(timespec='seconds'),
    'canonical': {'checkpoint': reg['checkpoint'], 'preprocess': reg['preprocess_version'],
                  'status': reg['status']},
    'registry_consumers': count_registry_consumers(),
    'n_registry_consumers': len(count_registry_consumers()),
    'hardcoded_checkpoint_refs': count_hardcoded(),
    'tests': count_tests(),
    'lineage': {'train_groups': lineage['train_groups'], 'val_groups': lineage['val_groups'],
                'cross_split': len(lineage['cross_split_groups']),
                'unparsed': lineage['n_unparsed']},
    'snapshot': {'tag': 'planner-v1.0-audit-fixed-rc1', 'tracked_files': snap_files},
    'deployment_eval_headline': {
        'q_mean_mrad': dep['q']['mean'], 'q_ci95': dep['q']['ci95'],
        'fk_mean_mm': dep['fk']['mean'],
        'gain_vx': dep['gains']['vx'].get('gain'),
        'gain_vyaw': dep['gains']['vyaw'].get('gain'),
        'verdict': 'see paired_ab_result.json (v1.1 vs RC1 on same 414 val clips)',
    },
}
out = REPO / 'humanoidverse/data/bdx_planner_v2combo/closure_status.json'
out.write_text(json.dumps(status, indent=1))
print(json.dumps(status, indent=1))
