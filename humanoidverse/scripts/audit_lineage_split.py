"""Lineage-level split audit — persists the zero-leakage result as a
reproducible artifact (audit 2026-08-24 requirement)."""
import json, re, sys, datetime
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/tcl/Desktop/start/dataset')
SPLITS = Path(__file__).parents[1] / 'data' / 'bdx_planner_v2combo' / 'splits.json'

def lineage_key(ver, c):
    gid, parent, cid = c.get('group_id'), c.get('parent_clip'), c['id']
    base = re.sub(r'_(mirror|speed[\d.]+|rev)$', '', cid)
    if gid and str(gid) not in ('None', ''):
        return f'{ver}:grp:{gid}'
    if parent and str(parent) not in ('None', ''):
        return f'{ver}:par:{parent}'
    return f'{ver}:base:{base}'

def main():
    sp = json.loads(SPLITS.read_text())
    lineage = {}
    unparsed = []
    for ver in sorted(p.name for p in ROOT.iterdir()
                      if (p / 'manifest.json').exists()):
        for c in json.load(open(ROOT / ver / 'manifest.json'))['clips']:
            lineage[f'{ver}/{c["id"]}'] = lineage_key(ver, c)
    def resolve(n):
        # merged-pool names carry synthetic prefixes -> map back to versions
        if n.startswith('v5i/'):
            n2 = 'v5_intervene2/' + n[4:]
            return n2 if n2 in lineage else n
        if n.startswith('v6p/'):
            n2 = 'v6_patch/' + n[4:]
            return n2 if n2 in lineage else n
        return n

    tr = defaultdict(set); va = defaultdict(set)
    for n in sp['train']:
        m = resolve(n)
        if m in lineage: tr[lineage[m]].add(n)
        else: unparsed.append(n)
    for n in sp['val']:
        m = resolve(n)
        if m in lineage: va[lineage[m]].add(n)
        else: unparsed.append(n)
    overlap = sorted(k for k in va if k in tr)
    result = {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'splits_file': str(SPLITS),
        'lineage_rules': 'group_id > parent_clip > augmentation-stripped base id',
        'train_groups': len(tr), 'val_groups': len(va),
        'cross_split_groups': overlap,
        'val_clips_in_leaked_groups': sum(len(va[k]) for k in overlap),
        'unparsed_names': unparsed[:20], 'n_unparsed': len(unparsed),
    }
    out = SPLITS.parent / 'lineage_audit_result.json'
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'cross_split_groups'}, indent=1))
    print(f'cross_split_groups: {len(overlap)} -> {out}')

if __name__ == '__main__':
    main()
