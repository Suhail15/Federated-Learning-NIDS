"""Prepare immutable real splits, training-fitted scale and fit provenance."""
import argparse
import platform
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .common import (ATTACK_MAP, FEATURES, ARMS, SEEDS, read_raw, group_split, client_assignments,
                     assert_boundaries, csv_write, dump, sha)


def prepare(train_path, test_path, evidence, local):
    evidence, local = Path(evidence), Path(local)
    if (evidence / 'protocol.yaml').exists():
        raise ValueError('Use a new experiment directory; frozen inputs must not be overwritten')
    evidence.mkdir(parents=True, exist_ok=True); local.mkdir(parents=True, exist_ok=True)
    train, test = read_raw(train_path), read_raw(test_path)
    groups = {}
    for i, key in enumerate(train['fp']):
        groups.setdefault(key, []).append(i)
    conflicting = {key for key, members in groups.items() if len(set(train['y'][members])) > 1}
    conflict_mask = np.isin(train['fp'], list(conflicting))
    conflict_ids = train['ids'][conflict_mask].tolist()
    overlap = np.isin(train['fp'], test['fp'])
    original_count = len(train['y'])
    exclusions = train['ids'][overlap].tolist()
    original_counts = np.bincount(train['y'], minlength=5).tolist()
    for k in ['x', 'y', 'ids', 'fp']:
        train[k] = train[k][~(overlap | conflict_mask)]
    ti, vi = group_split(train['y'], train['fp'])
    train_clients = client_assignments(train['y'][ti], train['fp'][ti])
    val_clients = client_assignments(train['y'][vi], train['fp'][vi])
    scaler = MinMaxScaler().fit(train['x'][ti])
    audit = assert_boundaries(train['ids'][ti], train['ids'][vi], test['ids'], train['ids'][ti],
                             train_clients, train['fp'][ti], train['fp'][vi], test['fp'])
    audit.update({'excluded_development_test_overlap': len(exclusions),
                  'excluded_record_ids': exclusions, 'conflicting_groups': len(conflicting), 'conflicting_record_ids': conflict_ids, 'training_fit_record_count': len(ti),
                  'test_size': len(test['ids']), 'validation_real_only': True, 'test_real_only': True})
    for c in range(2):
        if set(train['y'][ti][train_clients == c]) != set(range(5)):
            raise ValueError('Every training client must contain all five classes')
    # Training never loads test.npz. Only the final evaluator opens it.
    np.savez_compressed(local / 'development.npz', x_train=scaler.transform(train['x'][ti]).astype('float32'),
                        y_train=train['y'][ti], train_ids=train['ids'][ti], clients=train_clients,
                        x_val=scaler.transform(train['x'][vi]).astype('float32'), y_val=train['y'][vi],
                        val_ids=train['ids'][vi], val_clients=val_clients)
    np.savez_compressed(local / 'test.npz', x=scaler.transform(test['x']).astype('float32'),
                        y=test['y'], ids=test['ids'])
    preprocessing = {'features': FEATURES, 'data_min': scaler.data_min_.tolist(),
                     'data_max': scaler.data_max_.tolist(), 'scale': scaler.scale_.tolist(),
                     'min': scaler.min_.tolist(), 'fit_ids': train['ids'][ti].tolist(),
                     'fit_source_sha256': train['sha'], 'clip_real_holdout': False}
    dump(evidence / 'preprocessing.json', preprocessing)
    dump(evidence / 'attack-map.json', {'mapping': ATTACK_MAP, 'httptunnel_family': 'U2R',
         'taxonomy_note': 'Explicit five-family convention; httptunnel placement differs across published tables. Never selected by test performance.',
         'taxonomy_reference': 'https://aircconline.com/ijnsa/V12N4/12420ijnsa02.pdf'})
    dump(evidence / 'schema.json', {'features': FEATURES, 'raw_columns': 43, 'difficulty_excluded': True,
         'binary': ['land', 'logged_in'], 'integer': ['duration', 'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count'],
         'rates': [f for f in FEATURES if f not in ['land', 'logged_in', 'duration', 'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']],
         'synthetic_projection': 'inverse scale; round binary/integer; clip rates [0,1] and integer values >=0; reapply training scale'})
    dump(evidence / 'provenance.json', {'source': 'Existing local nsl-kdd files; original acquisition date/URL unrecorded',
         'independent_origin_verified': False, 'reference': 'https://www.unb.ca/cic/datasets/nsl.html',
         'train': {'filename': Path(train_path).name, 'sha256': train['sha'], 'rows': original_count, 'class_counts': original_counts},
         'test': {'filename': Path(test_path).name, 'sha256': test['sha'], 'rows': len(test['y']), 'class_counts': np.bincount(test['y'], minlength=5).tolist()},
         'excluded_development_records': int((overlap | conflict_mask).sum()), 'conflicting_group_policy': 'quarantine every development record in groups with conflicting family labels; never relabel', 'test_preserved_complete': True})
    audit['reduced_feature_collisions_train_validation'] = len(set(map(tuple, train['x'][ti])) & set(map(tuple, train['x'][vi])))
    audit['reduced_feature_collisions_development_test'] = len(set(map(tuple, train['x'])) & set(map(tuple, test['x'])))
    dump(evidence / 'leakage-audit.json', audit)
    records = []
    for role, idx, assignment in [('train', ti, train_clients), ('validation', vi, val_clients)]:
        for i, c in zip(idx, assignment):
            records.append({'record_id': train['ids'][i], 'source_file_sha256': train['sha'], 'source_row': int(str(train['ids'][i]).rsplit(':', 1)[1]),
                            'feature_fingerprint': train['fp'][i], 'role': role, 'client_id': int(c)})
    for i, rid in enumerate(test['ids']):
        records.append({'record_id': rid, 'source_file_sha256': test['sha'], 'source_row': i+1,
                        'feature_fingerprint': test['fp'][i], 'role': 'test', 'client_id': ''})
    csv_write(evidence / 'splits.csv', ['record_id', 'source_file_sha256', 'source_row', 'feature_fingerprint', 'role', 'client_id'], records)
    counts, steps = [], []
    for stage, y, clients in [('train', train['y'][ti], train_clients), ('validation', train['y'][vi], val_clients)]:
        for client in range(2):
            n = np.bincount(y[clients == client], minlength=5)
            for label in range(5):
                counts.append(dict(stage=stage, client_id=client, label=label, real_count=int(n[label]), synthetic_count=0, duplicate_count=0))
            if stage == 'train':
                expanded = int(n.sum() + sum(max(0, int(n[1]-n[c])) for c in [2, 3, 4]))
                steps.append(int(np.ceil(expanded / 128)))
    csv_write(evidence / 'class-counts.csv', ['stage', 'client_id', 'label', 'real_count', 'synthetic_count', 'duplicate_count'], counts)
    protocol = {'version': 1, 'arms': ARMS, 'seeds': SEEDS, 'split_seed': 42, 'partition_seed': 42,
                'rounds': 5, 'batch_size': 128, 'steps_per_client_per_round': steps,
                'learning_rate': .0001, 'gan_epochs': 50, 'gan_batches': {'2': 512, '3': 32, '4': 2},
                'gan_scope': 'pooled original training only; independent class generators',
                'aggregation_weights': 'original real training counts', 'execution': 'serial in-process clients with Flower FedAvg.aggregate_fit; no network transport',
                'checkpoint_selection': 'final round; no validation/test selection',
                'holdout': 'all locally held KDDTest+ records; file origin unverified',
                'test_sha256': sha(local / 'test.npz'), 'development_sha256': sha(local / 'development.npz'),
                'preprocessing_sha256': sha(evidence / 'preprocessing.json')}
    # JSON is valid YAML 1.2; no additional YAML dependency is required.
    dump(evidence / 'protocol.yaml', protocol)
    print(f'Prepared real train={len(ti)}, validation={len(vi)}, test={len(test["y"])}; excluded={len(exclusions)}; client steps={steps}', flush=True)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True); p.add_argument('--test', required=True)
    p.add_argument('--evidence', required=True); p.add_argument('--local', required=True)
    a = p.parse_args(); prepare(a.train, a.test, a.evidence, a.local)


if __name__ == '__main__':
    main()
