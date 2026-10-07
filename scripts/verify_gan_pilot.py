"""Independently recompute validation metrics and audit the pilot's fit boundary."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def load(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def table(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def verify(root):
    p = load(root/'plan.json'); seal = load(root/'evaluation-seal.json')
    assert digest(root/'splits.csv') == p['splits_sha256']
    assert digest(root/'preprocessing.json') == p['preprocessing_sha256']
    split = table(root/'splits.csv'); ids = [r['record_id'] for r in split]
    assert len(ids) == len(set(ids))
    train = {r['record_id']:r for r in split if r['role'] == 'core-training'}
    mixed = {r['record_id']:r for r in split if r['role'] == 'original-validation'}
    hard = {r['record_id']:r for r in split if r['challenge_member'] == '1'}
    assert not (set(train)&set(hard) or set(train)&set(mixed))
    assert not ({r['feature_fingerprint'] for r in train.values()} &
                {r['feature_fingerprint'] for r in split if r['role'] != 'core-training'})
    assert all(r['subtype'] not in p['held_subtypes'] for r in train.values())
    assert set(load(root/'preprocessing.json')['fit_ids']) == set(train)
    assert not seal['test_read'] and len(seal['runs']) == 15
    assert {(r['arm'],r['seed']) for r in seal['runs']} == {(a,s) for a in p['arms'] for s in p['seeds']}
    for seed in p['seeds']:
        out = root/'augmentation'/f'seed-{seed}'; m = load(out/'manifest.json')
        fit = table(out/'fit-ids.csv')
        assert digest(out/'fit-ids.csv') == m['fit_ids_sha256']
        assert {r['record_id'] for r in fit} == {k for k,r in train.items() if int(r['family']) in (2,3,4)}
        assert not m['validation_used_to_fit_or_filter'] and m['exact_core_synthetic_pattern_overlap'] == 0
        if p['gan'].get('family_balanced_fit',False):
            lineage = table(out/'gan-fit-lineage.csv')
            assert digest(out/'gan-fit-lineage.csv') == m['fit_lineage_sha256']
            assert {r['source_record_id'] for r in lineage} == {r['record_id'] for r in fit}
            assert len(lineage) == p['gan']['fit_rows_including_bootstrap'] == m['fit_rows_including_bootstrap']
            counts = [sum(int(r['family']) == k for r in lineage) for k in (2,3,4)]
            assert len(set(counts)) == 1
            assert all(r['family'] == train[r['source_record_id']]['family'] for r in lineage)
        sources = table(out/'ros-lineage.csv'); counts = [[0]*5 for _ in range(2)]; low = [[0]*5 for _ in range(2)]
        for r in sources:
            source = train[r['source_record_id']]
            assert source['client_id'] == r['client_id'] and source['family'] == r['family']
            c,label = int(r['client_id']),int(r['family']); counts[c][label] += 1
            low[c][label] += int(r['in_025'])
        assert counts == p['quotas']['100'] and low == p['quotas']['025']
        for r in m['sampling_audit']:
            assert r['accepted'] == sum(q[r['family']] for q in p['quotas']['100'])
            assert r['accepted']+r['wrong_label']+r['nonfinite']+r['core_duplicate']+r['generated_duplicate']+r['surplus_valid'] == r['requested']
    scores,initials,schedules = {},{},{}
    for run in seal['runs']:
        arm,seed = run['arm'],run['seed']; out = root/arm/f'seed-{seed}'
        m = load(out/'manifest.json'); saved = load(out/'metrics.json')
        assert digest(out/'manifest.json') == run['manifest_sha256']
        assert m['status'] == 'completed' and m['completed_rounds'] == 5 and m['client_failures'] == 0
        assert not m['test_evaluated'] and m['optimizer_updates'] == 5*sum(p['steps_per_client_per_round'])
        assert m['plan_sha256'] == digest(root/'plan.json') and m['checkpoint_sha256'] == run['checkpoint_sha256']
        initials.setdefault(seed,m['initial_sha256']); assert initials[seed] == m['initial_sha256']
        history = table(out/'training.csv'); assert len(history) == 10
        for r in history:
            key = (seed,r['round'],r['client_id'])
            schedules.setdefault(key,r['label_schedule_sha256']); assert schedules[key] == r['label_schedule_sha256']
        predictions = table(out/'predictions.csv')
        assert len(predictions) == len(hard)+len(mixed)
        assert {r['split'] for r in predictions} == {'subtype-challenge','mixed-validation'}
        for name,expected in [('subtype-challenge',hard),('mixed-validation',mixed)]:
            matrix = [[0]*5 for _ in range(5)]; seen = set(); log_loss = 0
            for r in predictions:
                if r['split'] != name: continue
                rid = r['record_id']; assert rid in expected and rid not in seen; seen.add(rid)
                y,pred = int(r['y_true']),int(r['y_pred'])
                assert y == int(expected[rid]['family']) and r['subtype'] == expected[rid]['subtype']
                prob = [float(r[f'p{i}']) for i in range(5)]
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in prob) and abs(sum(prob)-1) < 1e-5
                assert max(range(5),key=lambda i:prob[i]) == pred
                matrix[y][pred] += 1; log_loss -= math.log(max(1e-7,min(1,prob[y])))
            assert seen == set(expected) and matrix == saved[name]['confusion_matrix']
            support = [sum(r) for r in matrix]; predicted = [sum(r[i] for r in matrix) for i in range(5)]
            recall = [matrix[i][i]/support[i] for i in range(5)]
            precision = [matrix[i][i]/predicted[i] if predicted[i] else 0 for i in range(5)]
            f1 = [2*matrix[i][i]/(support[i]+predicted[i]) if support[i]+predicted[i] else 0 for i in range(5)]
            recomputed = dict(macro_f1=statistics.mean(f1),accuracy=sum(matrix[i][i] for i in range(5))/len(expected),
                balanced_accuracy=statistics.mean(recall),normal_false_positive_rate=1-recall[0],
                binary_attack_recall=1-sum(matrix[i][0] for i in range(1,5))/sum(support[1:]),loss=log_loss/len(expected))
            for metric,value in recomputed.items():
                assert abs(value-saved[name][metric]) < (1e-5 if metric == 'loss' else 1e-12)
            assert support == saved[name]['support']
            for metric,values in [('precision',precision),('recall',recall),('f1',f1)]:
                assert all(abs(a-b)<1e-12 for a,b in zip(values,saved[name][metric]))
            scores[(arm,seed,name)] = recomputed
    comparison = load(root/'comparison.json')
    for dose,controls in comparison['gan_minus_control'].items():
        passes = []
        for control,v in controls.items():
            gan = f'balanced-ctgan-{dose}'
            for name in ['subtype-challenge','mixed-validation']:
                for metric in ['macro_f1','normal_false_positive_rate']:
                    delta = [scores[(gan,s,name)][metric]-scores[(control,s,name)][metric] for s in p['seeds']]
                    obj = v[name][metric]
                    assert all(abs(a-b)<1e-12 for a,b in zip(delta,obj['values']))
                    assert abs(statistics.mean(delta)-obj['mean']) < 1e-12
                    assert abs(statistics.stdev(delta)-obj['std']) < 1e-12
                    assert sum(x>0 for x in delta) == obj['positive_seeds']
            a,b = v['subtype-challenge'],v['mixed-validation']
            ok = (a['macro_f1']['mean'] >= .01 and a['macro_f1']['positive_seeds'] >= 2 and
                  b['macro_f1']['mean'] >= -.01 and a['normal_false_positive_rate']['mean'] <= .01 and
                  b['normal_false_positive_rate']['mean'] <= .01)
            assert ok == v['meets_predeclared_rule']; passes.append(ok)
        assert all(passes) == comparison['meets_pilot_rule'][dose]
    result = dict(runs_verified=15,ctgan_fits_verified=3,core_training_records=len(train),
        challenge_records_per_run=len(hard),mixed_validation_records_per_run=len(mixed),
        complete_unique_predictions=True,independent_metric_recomputation='passed',
        fit_boundaries='passed',withheld_subtypes_excluded_from_fits=True,client_only_ros_lineage='passed',
        paired_initial_weights=True,paired_balanced_family_schedules=True,matched_update_budgets=True,
        pilot_decision_recomputed=True,test_re_evaluated=False,scope='exploratory validation only')
    result['GAN_fit_bootstrap_lineage_verified'] = bool(p['gan'].get('family_balanced_fit',False))
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--evidence',required=True,type=Path)
    print(json.dumps(verify(ap.parse_args().evidence),indent=2))
