"""Independently recompute held-out metrics and check all predeclared runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(p): return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--evidence',required=True)
    root=Path(ap.parse_args().evidence); protocol=load(root/'protocol.yaml'); seal=load(root/'evaluation-seal.json')
    with (root/'splits.csv').open() as f: split=list(csv.DictReader(f))
    train={r['record_id'] for r in split if r['role']=='train'}
    val={r['record_id'] for r in split if r['role']=='validation'}
    test={r['record_id'] for r in split if r['role']=='test'}
    assert not (train&val or train&test or val&test)
    assert set(load(root/'preprocessing.json')['fit_ids'])==train
    assert len(seal['runs'])==12
    scores={}
    for seed in protocol['seeds']:
        a=root/'augmentation'/f'seed-{seed}'
        with (a/'fit-ids.csv').open() as f: fit=list(csv.DictReader(f))
        assert {r['record_id'] for r in fit} <= train
        with (a/'ros-lineage.csv').open() as f: sources=list(csv.DictReader(f))
        owners={r['record_id']:r['client_id'] for r in split if r['role']=='train'}
        assert all(r['source_record_id'] in train and owners[r['source_record_id']]==r['client_id'] for r in sources)
    initials={}
    for run in seal['runs']:
        arm,seed=run['arm'],run['seed']; out=root/arm/f'seed-{seed}'
        manifest=load(out/'manifest.json'); saved=load(out/'metrics.json')
        assert manifest['status']=='completed' and manifest['completed_rounds']==5
        assert manifest['optimizer_updates']==5*sum(protocol['steps_per_client_per_round'])
        assert hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()==run['manifest_sha256']
        initials.setdefault(seed,manifest['initial_sha256']); assert initials[seed]==manifest['initial_sha256']
        matrix=[[0]*5 for _ in range(5)]; seen=set()
        with (out/'predictions.csv').open() as f:
            for row in csv.DictReader(f):
                if row['split']!='real-test':
                    assert row['record_id'] in val
                    continue
                rid=row['record_id']; assert rid in test and rid not in seen; seen.add(rid)
                truth,pred=int(row['y_true']),int(row['y_pred'])
                prob=[float(row[f'p{i}']) for i in range(5)]
                assert all(math.isfinite(v) for v in prob)
                assert abs(sum(prob)-1)<1e-5 and max(range(5),key=lambda i:prob[i])==pred
                matrix[truth][pred]+=1
        assert seen==test and matrix==saved['confusion_matrix']
        supports=[sum(r) for r in matrix]; predicted=[sum(r[i] for r in matrix) for i in range(5)]
        f1=[2*matrix[i][i]/(supports[i]+predicted[i]) if supports[i]+predicted[i] else 0 for i in range(5)]
        score=statistics.mean(f1); accuracy=sum(matrix[i][i] for i in range(5))/len(test)
        balanced=statistics.mean(matrix[i][i]/supports[i] for i in range(5))
        fpr=(supports[0]-matrix[0][0])/supports[0]
        assert abs(score-saved['macro_f1'])<1e-12 and abs(accuracy-saved['accuracy'])<1e-12
        assert abs(balanced-saved['balanced_accuracy'])<1e-12 and abs(fpr-saved['normal_false_positive_rate'])<1e-12
        scores[(arm,seed)]=score
    comparison=load(root/'comparison.json')
    for control in ['fedavg-real','fedavg-ros']:
        delta=[scores[('fedavg-gan',s)]-scores[(control,s)] for s in protocol['seeds']]
        assert abs(statistics.mean(delta)-comparison['gan_minus_control'][control]['mean'])<1e-12
    output={'runs_verified':12,'test_records_per_run':len(test),'training_records':len(train),'validation_records':len(val),
            'prediction_coverage':'complete and unique','independent_metric_recomputation':'passed',
            'paired_initial_weights':'identical within each seed','matched_update_counts':'passed',
            'GAN_fit_IDs_training_only':True,'oversampling_lineage_training_and_client_only':True}
    (root/'independent-verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
