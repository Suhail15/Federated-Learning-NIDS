"""Describe omitted raw features on real core training; never score a model."""
import argparse
import csv
from pathlib import Path
import numpy as np
from benchmark.common import ATTACK_MAP, CLASSES, dump, sha

FIELDS = {'src_bytes':4, 'dst_bytes':5, 'num_failed_logins':10, 'num_compromised':12,
          'root_shell':13, 'su_attempted':14, 'num_root':15, 'num_file_creations':16,
          'num_shells':17, 'num_access_files':18}


def audit(source, local, output):
    digest = sha(source)
    with np.load(local/'development.npz') as d:
        ids,y = d['core_ids'],d['y_core']
    if any(str(rid).rsplit(':',1)[0] != digest for rid in ids):
        raise ValueError('Development source changed')
    with source.open(newline='') as f:
        rows = list(csv.reader(f))
    indices = np.array([int(str(rid).rsplit(':',1)[1])-1 for rid in ids])
    actual = np.array([ATTACK_MAP[rows[i][41].strip().rstrip('.')] for i in indices])
    if not np.array_equal(y,actual):
        raise ValueError('Raw source family labels changed')
    stats = {}
    for name,col in FIELDS.items():
        values = np.array([float(rows[i][col]) for i in indices]); stats[name] = {}
        if not np.isfinite(values).all(): raise ValueError('Nonfinite omitted feature')
        for label,family in enumerate(CLASSES):
            v = values[y == label]
            stats[name][family] = dict(real_core_rows=len(v),positive_rows=int(np.sum(v>0)),
                positive_fraction=float(np.mean(v>0)),mean=float(v.mean()),max=float(v.max()))
    dump(output,dict(scope='descriptive real core training statistics only, not detection-performance evidence',
        source_sha256=digest,core_development_sha256=sha(local/'development.npz'),
        fields_present_in_raw_but_absent_from_current_23_feature_model=stats,
        selection_evidence='notebooks/clientmk3i.ipynb drops these fields; benchmark/common.py preserves the 23-feature selection',
        hypothesis='Globally sparse attack-specific features may contain information important to rare families. A generator operating on the reduced feature set cannot restore omitted information. Restore and validate these fields in a separate locked comparison; no performance claim is established by this descriptive audit.'))
    print(f'Wrote descriptive audit of {len(FIELDS)} omitted fields on {len(y):,} real core rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-development',required=True,type=Path)
    ap.add_argument('--local',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    a = ap.parse_args(); audit(a.raw_development,a.local,a.output)
