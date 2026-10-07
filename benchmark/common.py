"""Shared schema, provenance, metrics and JSON/CSV exports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

FEATURES = ['duration', 'land', 'hot', 'logged_in', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']
INDICES = [0, 6, 9, 11, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
ATTACKS = {
    0: ['normal'],
    1: ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'mailbomb', 'worm'],
    2: ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'],
    3: ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'sendmail', 'snmpgetattack', 'snmpguess', 'named', 'xlock', 'xsnoop'],
    4: ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 'ps', 'sqlattack', 'xterm'],
}
ATTACK_MAP = {name: label for label, names in ATTACKS.items() for name in names}
ARMS = ['centralized', 'fedavg-real', 'fedavg-ros', 'fedavg-gan']
SEEDS = [11, 22, 33]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def csv_write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_raw(path):
    """Parse 41 raw features plus attack/difficulty; hash full feature tuples."""
    digest = sha(path)
    x, y, ids, fingerprints = [], [], [], []
    with open(path, newline='') as f:
        for number, row in enumerate(csv.reader(f), 1):
            if len(row) != 43:
                raise ValueError(f'Raw record {number} must contain 43 columns')
            attack = row[41].strip().rstrip('.')
            if attack not in ATTACK_MAP:
                raise ValueError(f'Unmapped attack: {attack}')
            values = np.array([float(row[i]) for i in INDICES], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite raw features')
            # Normalize numeric spellings; preserve the three nominal feature tokens.
            canonical = [v if i in (1, 2, 3) else format(float(v), '.17g') for i, v in enumerate(row[:41])]
            fingerprints.append(hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest())
            x.append(values); y.append(ATTACK_MAP[attack]); ids.append(f'{digest}:{number}')
    return {'x': np.asarray(x), 'y': np.asarray(y, dtype=np.int64),
            'ids': np.asarray(ids), 'fp': np.asarray(fingerprints), 'sha': digest}


def group_split(y, fp, validation_fraction=.2, seed=42):
    """Stratify full-feature groups, never individual copies of one group."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, key in enumerate(fp):
        groups.setdefault(key, []).append(i)
    for members in groups.values():
        if len(set(y[members])) != 1:
            raise ValueError('Conflicting family labels for identical raw features')
    validation = []
    for label in range(5):
        keys = [k for k, v in groups.items() if y[v[0]] == label]
        if len(keys) < 2:
            raise ValueError('Each class needs at least two distinct feature groups')
        rng.shuffle(keys)
        target = int(round(np.sum(y == label) * validation_fraction))
        selected, count = [], 0
        for key in keys:
            if count < target and len(selected) < len(keys) - 1:
                selected.append(key); count += len(groups[key])
        validation.extend(i for key in selected for i in groups[key])
    val = np.asarray(sorted(validation), dtype=np.int64)
    train = np.setdiff1d(np.arange(len(y)), val)
    return train, val


def client_assignments(y, fp, seed=42):
    """Assign whole groups by class to the least populated of two clients."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, key in enumerate(fp):
        groups.setdefault(key, []).append(i)
    clients = np.empty(len(y), dtype=np.int64)
    for label in range(5):
        keys = [k for k, v in groups.items() if y[v[0]] == label]
        rng.shuffle(keys)
        counts = [0, 0]
        for key in keys:
            client = int(np.argmin(counts))
            clients[groups[key]] = client; counts[client] += len(groups[key])
    return clients


def assert_boundaries(train_ids, val_ids, test_ids, fit_ids, clients, train_fp=None, val_fp=None, test_fp=None):
    sets = [set(train_ids), set(val_ids), set(test_ids)]
    if any(len(s) != len(a) for s, a in zip(sets, [train_ids, val_ids, test_ids])):
        raise ValueError('Repeated source record IDs')
    if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('Evaluation records overlap training')
    if set(fit_ids) != sets[0]:
        raise ValueError('Transform fit IDs must equal real training IDs')
    if len(clients) != len(train_ids) or not set(clients) <= {0, 1}:
        raise ValueError('Client assignments do not cover training exactly')
    if train_fp is not None:
        fps = [set(train_fp), set(val_fp), set(test_fp)]
        if any(fps[i] & fps[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError('Full-feature groups overlap evaluation')
        owners = {}
        for key, client in zip(train_fp, clients):
            if key in owners and owners[key] != client:
                raise ValueError('A full-feature group spans clients')
            owners[key] = client
    return {'record_overlap': 0, 'fit_contains_holdout': False,
            'client_partition_complete': True, 'full_feature_overlap': 0}


def metrics(y, probabilities):
    if probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid evaluation probabilities')
    if set(y) != set(range(5)):
        raise ValueError('Evaluation requires support for all five classes')
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=np.arange(5), zero_division=0)
    cm = confusion_matrix(y, prediction, labels=np.arange(5))
    loss = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-7, 1)).mean()
    return {'accuracy': float(accuracy_score(y, prediction)), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean()), 'loss': float(loss),
            'normal_false_positive_rate': float(np.mean(prediction[y == 0] != 0)),
            'binary_attack_recall': float(np.mean(prediction[y != 0] != 0)),
            'support': support.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'f1': f1.tolist(), 'never_predicted': [i for i in range(5) if not np.any(prediction == i)],
            'confusion_matrix': cm.tolist(), 'zero_division': 0, 'label_order': list(range(5))}
