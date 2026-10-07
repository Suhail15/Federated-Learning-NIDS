"""Training-only conditional GAN pilot, evaluated on development data only.

The KDDTest+ file and v1 test.npz are deliberately absent from this interface.
See README.md in this directory for the predeclared comparisons and limits.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from benchmark.common import (FEATURES, INDICES, CLASSES, SEEDS, csv_write,
                              dump, load, metrics, read_raw, sha)

HELD_SUBTYPES = ['back', 'portsweep', 'warezclient', 'loadmodule']
ARMS = ['balanced-real', 'balanced-ros-025', 'balanced-ctgan-025',
        'balanced-ros-100', 'balanced-ctgan-100']
DOSES = {'025': .25, '100': 1.0}
INTEGERS = ['duration', 'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
BINARY = ['land', 'logged_in']
SPLITS = ['subtype-challenge', 'mixed-validation']


def source_hash():
    h = hashlib.sha256()
    for p in sorted(list(Path('gan_research').glob('*.py')) +
                    [Path('model.py'), Path('benchmark/common.py'), Path('benchmark/run.py'),
                     Path('requirements-gan.txt')]):
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def subtype_mask(subtypes, fingerprints, held=HELD_SUBTYPES):
    """Move entire full-feature groups if any member has a withheld subtype."""
    groups = set(fingerprints[np.isin(subtypes, held)])
    return np.isin(fingerprints, list(groups))


def quotas(y, clients, dose):
    return [[int(math.ceil(np.sum((clients == c) & (y == label)) * dose))
             if label in (2, 3, 4) else 0 for label in range(5)] for c in range(2)]


def project(raw, real_class):
    """Apply training-derived constants and the original feature domains."""
    out = np.asarray(raw, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != len(FEATURES) or not np.isfinite(out).all():
        raise ValueError('Invalid synthetic feature matrix')
    for i, name in enumerate(FEATURES):
        if name in BINARY:
            out[:, i] = (out[:, i] >= .5).astype(float)
        elif name in INTEGERS:
            out[:, i] = np.maximum(0, np.rint(out[:, i]))
        else:
            out[:, i] = np.round(np.clip(out[:, i], 0, 1), 2)
        if np.all(real_class[:, i] == real_class[0, i]):
            out[:, i] = real_class[0, i]
    return out


def batch_plan(x, y, seed, steps, batch=128):
    """Uniform family exposure with an identical label schedule across arms."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(5), math.ceil(steps * batch / 5))[:steps * batch]
    rng.shuffle(labels)
    index = np.empty(len(labels), dtype=np.int64)
    for label in range(5):
        candidates = np.flatnonzero(y == label)
        if not len(candidates):
            raise ValueError('Each client must have real support for every family')
        selected = labels == label
        index[selected] = rng.choice(candidates, int(selected.sum()), replace=True)
    return x[index], labels


def prepare(raw_path, previous, prior_local, evidence, local):
    if (evidence / 'plan.json').exists() or (local / 'development.npz').exists():
        raise ValueError('An existing experiment cannot be overwritten')
    old = load(previous / 'protocol.yaml')
    if sha(prior_local / 'development.npz') != old['development_sha256']:
        raise ValueError('Prior development array changed')
    raw = read_raw(raw_path)  # Only the original development source is read.
    with raw_path.open(newline='') as f:
        subtypes = np.array([r[41].strip().rstrip('.') for r in csv.reader(f)])
    with np.load(prior_local / 'development.npz') as d:
        train_ids, val_ids, clients = d['train_ids'], d['val_ids'], d['clients']
        if any(str(rid).rsplit(':', 1)[0] != raw['sha'] for rid in np.r_[train_ids, val_ids]):
            raise ValueError('Original development source does not match the v1 IDs')
        tr = np.array([int(str(rid).rsplit(':', 1)[1]) - 1 for rid in train_ids])
        va = np.array([int(str(rid).rsplit(':', 1)[1]) - 1 for rid in val_ids])
        if not (np.array_equal(d['y_train'], raw['y'][tr]) and np.array_equal(d['y_val'], raw['y'][va])):
            raise ValueError('Source family labels changed')
    mask = subtype_mask(subtypes[tr], raw['fp'][tr])
    core, novel = tr[~mask], tr[mask]
    cc = clients[~mask]
    if set(raw['fp'][core]) & (set(raw['fp'][novel]) | set(raw['fp'][va])):
        raise ValueError('Full-feature group leakage')
    if np.isin(subtypes[core], HELD_SUBTYPES).any():
        raise ValueError('Withheld attack subtype remains in a fit pool')
    owners = {}
    for fp, client in zip(raw['fp'][core], cc):
        if fp in owners and owners[fp] != client:
            raise ValueError('A full-feature group spans clients')
        owners[fp] = client
    scaler = MinMaxScaler().fit(raw['x'][core])
    hard = np.r_[va[raw['y'][va] == 0], novel]
    if set(raw['y'][hard]) != set(range(5)):
        raise ValueError('Subtype challenge requires support for all five families')
    local.mkdir(parents=True, exist_ok=True); evidence.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local / 'development.npz', raw_core=raw['x'][core],
                        x_core=scaler.transform(raw['x'][core]).astype('float32'),
                        y_core=raw['y'][core], core_ids=raw['ids'][core], clients=cc,
                        x_hard=scaler.transform(raw['x'][hard]).astype('float32'), y_hard=raw['y'][hard],
                        hard_ids=raw['ids'][hard], hard_subtypes=subtypes[hard],
                        x_mixed=scaler.transform(raw['x'][va]).astype('float32'), y_mixed=raw['y'][va],
                        mixed_ids=raw['ids'][va], mixed_subtypes=subtypes[va])
    dump(evidence / 'preprocessing.json', dict(features=FEATURES, scale=scaler.scale_.tolist(),
         min=scaler.min_.tolist(), data_min=scaler.data_min_.tolist(), data_max=scaler.data_max_.tolist(),
         fit_ids=raw['ids'][core].tolist(), clip_real_evaluation=False))
    rows = []
    for role, idx, cs in [('core-training', core, cc), ('subtype-validation', novel, [-1]*len(novel)),
                          ('original-validation', va, [-1]*len(va))]:
        for i, c in zip(idx, cs):
            rows.append(dict(record_id=raw['ids'][i], feature_fingerprint=raw['fp'][i],
                             family=int(raw['y'][i]), subtype=subtypes[i], role=role, client_id=int(c),
                             challenge_member=int(role == 'subtype-validation' or (role == 'original-validation' and raw['y'][i] == 0))))
    csv_write(evidence / 'splits.csv', list(rows[0]), rows)
    maxq = quotas(raw['y'][core], cc, 1.0)
    steps = [math.ceil((int(np.sum(cc == c)) + sum(maxq[c])) / 128) for c in range(2)]
    reduced_keys = lambda a: {tuple(row) for row in a}
    train_patterns = reduced_keys(raw['x'][core])
    plan = dict(version=1, experiment='conditional-gan-validation-v1', created_utc=utc(),
         status='locked-before-new-training', source_file=raw_path.name, source_sha256=raw['sha'],
         prior_development_sha256=old['development_sha256'], prior_evidence='nsl-kdd-v1',
         development_sha256=sha(local / 'development.npz'), preprocessing_sha256=sha(evidence/'preprocessing.json'),
         splits_sha256=sha(evidence/'splits.csv'), held_subtypes=HELD_SUBTYPES,
         seeds=SEEDS, arms=ARMS, doses=DOSES, batch_size=128, rounds=5, learning_rate=.0001,
         steps_per_client_per_round=steps, quotas={k:quotas(raw['y'][core],cc,v) for k,v in DOSES.items()},
         real_client_counts=[int(np.sum(cc == c)) for c in range(2)],
         training_class_counts=np.bincount(raw['y'][core], minlength=5).tolist(),
         client_class_counts=[np.bincount(raw['y'][core][cc == c], minlength=5).tolist() for c in range(2)],
         challenge_class_counts=np.bincount(raw['y'][hard], minlength=5).tolist(),
         mixed_validation_class_counts=np.bincount(raw['y'][va], minlength=5).tolist(),
         subtype_holdout_counts={s:int(np.sum(subtypes[novel] == s)) for s in HELD_SUBTYPES},
         reduced_pattern_overlap={'core_challenge':len(train_patterns & reduced_keys(raw['x'][hard])),
                                  'core_mixed_validation':len(train_patterns & reduced_keys(raw['x'][va]))},
         primary='subtype-challenge macro F1; deliberately altered class mixture', secondary='mixed-validation macro F1',
         exposure='class-balanced batches, same label schedules and optimizer-update budgets across arms',
         checkpoint_selection='final round, no checkpoint or epoch selection from evaluation',
         gan=dict(library='ctgan==0.11.0', torch='2.8.0', rdt='1.14.0', epochs=100,
                  embedding_dim=32, generator_dim=[128,128], discriminator_dim=[128,128], batch_size=500,
                  pac=10, log_frequency=True, cuda=False, generator_lr=.0002, discriminator_lr=.0002,
                  generator_decay=.000001, discriminator_decay=.000001, discriminator_steps=1,
                  discrete_columns=['family']+BINARY, fit_scope='pooled real core families 2,3,4 only',
                  conditional_label_filter=True, rate_precision=2, max_sample_attempts_per_family=30,
                  rejection='wrong conditional label, nonfinite, exact core pattern, exact generated duplicate',
                  constraints='domain projection and per-family constants from real core only'),
         pilot_success=dict(min_mean_primary_gain_vs_both_controls=.01, positive_seeds_vs_each_control=2,
                  max_mean_fpr_increase_vs_each_control=.01, max_mean_secondary_f1_drop_vs_each_control=.01,
                  controls=['balanced-real','matched-dose balanced-ros'], confidence='exploratory decision rule, not a significance test'),
         evaluation_boundary='development data only; KDDTest+ and test.npz never read by this pilot',
         prior_exposure='mixed validation was previously scored; subtype challenge was used in v1 training; neither is a new independent test',
         privacy='pooled preparation and serial in-process Flower clients, no private-silo claim')
    dump(evidence / 'plan.json', plan)
    print(json.dumps({k:plan[k] for k in ['training_class_counts','challenge_class_counts','steps_per_client_per_round','reduced_pattern_overlap']}, indent=2), flush=True)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen(evidence, local):
    p = load(evidence / 'plan.json')
    for path, key in [(local/'development.npz','development_sha256'),
                      (evidence/'preprocessing.json','preprocessing_sha256'), (evidence/'splits.csv','splits_sha256')]:
        if sha(path) != p[key]:
            raise ValueError(f'Frozen input changed: {path.name}')
    return p


def augment(evidence, local, seed):
    import torch
    from ctgan import CTGAN
    p = frozen(evidence, local); out = evidence/'augmentation'/f'seed-{seed}'
    pool_path = local/f'augmentation-{seed}.npz'
    if (out/'manifest.json').exists():
        m = load(out/'manifest.json')
        if m['code_sha256'] != source_hash() or sha(pool_path) != m['pool_sha256']:
            raise ValueError('Cannot reuse different-code augmentation')
        return
    torch.set_num_threads(4); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    d = np.load(local/'development.npz'); real, y = d['raw_core'], d['y_core']
    minority = np.isin(y, [2,3,4])
    frame = pd.DataFrame(real[minority], columns=FEATURES)
    for name in INTEGERS+BINARY:
        frame[name] = frame[name].astype('int64')
    frame['family'] = y[minority].astype(str)
    cfg = p['gan']
    model = CTGAN(embedding_dim=cfg['embedding_dim'], generator_dim=tuple(cfg['generator_dim']),
        discriminator_dim=tuple(cfg['discriminator_dim']), epochs=cfg['epochs'], batch_size=cfg['batch_size'],
        pac=cfg['pac'], log_frequency=cfg['log_frequency'], cuda=False, verbose=False)
    model.set_random_state(seed)
    print(f'CTGAN seed={seed} fit={len(frame)} real minority rows; epochs={cfg["epochs"]}', flush=True)
    started = time.monotonic(); model.fit(frame, discrete_columns=cfg['discrete_columns'])
    out.mkdir(parents=True, exist_ok=True)
    model.loss_values.to_csv(out/'losses.csv', index=False)
    model_path = local/f'ctgan-{seed}.pkl'; model.save(str(model_path))
    scale, offset = [np.asarray(load(evidence/'preprocessing.json')[key]) for key in ['scale','min']]
    forbidden = {tuple(row) for row in real}; accepted_all = set(); pools, diagnostics, audits = {}, [], []
    for label in [2,3,4]:
        total = sum(q[label] for q in p['quotas']['100'])
        accepted = []; counters = dict(requested=0, wrong_label=0, nonfinite=0, core_duplicate=0,
            generated_duplicate=0, projected=0, projected_by_feature=[0]*len(FEATURES), surplus_valid=0)
        for attempt in range(1, cfg['max_sample_attempts_per_family']+1):
            n = max(500, 2*(total-len(accepted)))
            sampled = model.sample(n, condition_column='family', condition_value=str(label))
            counters['requested'] += len(sampled)
            matching = sampled['family'].astype(str) == str(label)
            counters['wrong_label'] += int((~matching).sum())
            candidates = sampled.loc[matching, FEATURES].to_numpy(dtype=np.float64)
            finite = np.isfinite(candidates).all(axis=1)
            counters['nonfinite'] += int((~finite).sum()); candidates = candidates[finite]
            projected = project(candidates, real[y == label])
            changed = candidates != projected
            counters['projected'] += int(changed.any(axis=1).sum())
            counters['projected_by_feature'] = (np.asarray(counters['projected_by_feature'])+changed.sum(axis=0)).tolist()
            for row in projected:
                key = tuple(row)
                if key in forbidden:
                    counters['core_duplicate'] += 1
                elif key in accepted_all:
                    counters['generated_duplicate'] += 1
                elif len(accepted) < total:
                    accepted.append(row); accepted_all.add(key)
                else:
                    counters['surplus_valid'] += 1
            if len(accepted) == total:
                break
        if len(accepted) != total:
            raise ValueError(f'Conditional GAN cannot supply family {label}: {len(accepted)}/{total}; no ROS fallback')
        raw_fake = np.asarray(accepted)
        pools[f'class_{label}'] = (raw_fake*scale+offset).astype('float32')
        real_scaled = d['x_core'][y == label]
        for i, name in enumerate(FEATURES):
            diagnostics.append(dict(family=label,feature=name,real_count=int(np.sum(y == label)),synthetic_count=total,
                real_raw_mean=float(real[y == label,i].mean()),synthetic_raw_mean=float(raw_fake[:,i].mean()),
                scaled_wasserstein=float(wasserstein_distance(real_scaled[:,i],pools[f'class_{label}'][:,i])),
                training_constant=bool(np.all(real[y == label,i] == real[y == label,i][0]))))
        audits.append(dict(family=label,accepted=total,attempts=attempt,**counters))
    np.savez_compressed(pool_path, **pools)
    csv_write(out/'diagnostics.csv', list(diagnostics[0]), diagnostics)
    fit_rows = [dict(record_id=rid,family=int(label)) for rid,label in zip(d['core_ids'][minority],y[minority])]
    csv_write(out/'fit-ids.csv',list(fit_rows[0]),fit_rows)
    lineage = []
    for c in range(2):
        for label in [2,3,4]:
            ids = d['core_ids'][(d['clients'] == c)&(y == label)]
            rng = np.random.default_rng(seed*1000+c*100+label)
            for j,rid in enumerate(rng.choice(ids,p['quotas']['100'][c][label],replace=True)):
                lineage.append(dict(client_id=c,family=label,index=j,source_record_id=rid,
                                    in_025=int(j < p['quotas']['025'][c][label])))
    csv_write(out/'ros-lineage.csv',list(lineage[0]),lineage)
    dump(out/'manifest.json',dict(seed=seed,status='completed',code_sha256=source_hash(),
         plan_sha256=sha(evidence/'plan.json'),pool_sha256=sha(pool_path),model_sha256=sha(model_path),
         fit_ids_sha256=sha(out/'fit-ids.csv'),fit_count=int(minority.sum()),
         sampling_audit=audits,seconds=time.monotonic()-started,torch=torch.__version__,
         exact_core_synthetic_pattern_overlap=0,validation_used_to_fit_or_filter=False,
         pooled_training_only=True))
    print(f'CTGAN seed={seed} finished in {time.monotonic()-started:.1f}s; accepted={[a["accepted"] for a in audits]}', flush=True)


def client_pools(d, p, local, seed, arm):
    pools = []; aug = np.load(local/f'augmentation-{seed}.npz') if 'ctgan' in arm else None
    dose = arm.rsplit('-',1)[-1] if arm != 'balanced-real' else None
    for c in range(2):
        x, y = d['x_core'][d['clients'] == c], d['y_core'][d['clients'] == c]
        extra_x, extra_y = [], []
        if dose:
            for label in [2,3,4]:
                quantity = p['quotas'][dose][c][label]
                if aug is not None:
                    offset = 0 if c == 0 else p['quotas']['100'][0][label]
                    extra = aug[f'class_{label}'][offset:offset+quantity]
                else:
                    rng = np.random.default_rng(seed*1000+c*100+label)
                    extra = x[y == label][rng.choice(int(np.sum(y == label)),p['quotas']['100'][c][label],replace=True)[:quantity]]
                extra_x.append(extra); extra_y.append(np.full(quantity,label,dtype=np.int64))
        pools.append((np.concatenate([x]+extra_x),np.concatenate([y]+extra_y)))
    return pools


def train(evidence, local, seed, arm):
    import tensorflow as tf
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from benchmark.run import compile_dnn, initial, fitres
    p = frozen(evidence,local); out = evidence/arm/f'seed-{seed}'
    if (out/'manifest.json').exists():
        old = load(out/'manifest.json')
        if old['status'] == 'completed' and old['code_sha256'] == source_hash():
            return
        raise ValueError('Keep the existing incomplete/different-code run; use a new experiment ID')
    out.mkdir(parents=True,exist_ok=True); d = np.load(local/'development.npz')
    weights = initial(local,seed); tf.keras.backend.clear_session()
    models = [compile_dnn(seed,weights) for _ in range(2)]
    pools = client_pools(d,p,local,seed,arm)
    strategy = fl.server.strategy.FedAvg(initial_parameters=ndarrays_to_parameters(weights))
    checkpoint = local/arm/f'seed-{seed}'/'final.h5'; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    manifest = dict(arm=arm,seed=seed,status='running',started_utc=utc(),code_sha256=source_hash(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        plan_sha256=sha(evidence/'plan.json'),initial_sha256=sha(local/f'initial-{seed}.npz'),
        real_client_counts=p['real_client_counts'],pool_counts=[len(v[1]) for v in pools],
        optimizer_updates=p['rounds']*sum(p['steps_per_client_per_round']),
        tensorflow=tf.__version__,flower=fl.__version__,python=platform.python_version(),
        platform=platform.platform(),optimizer_state='local Adam slots persist across rounds',
        aggregation='Flower FedAvg.aggregate_fit with original real core counts',
        execution='two serial in-process clients',checkpoint_selection='final round',
        test_evaluated=False,command=' '.join(sys.argv))
    dump(out/'manifest.json',manifest); started = time.monotonic(); rows = []
    try:
        for rnd in range(1,p['rounds']+1):
            results = []
            for c,model in enumerate(models):
                model.set_weights(weights)
                bx,by = batch_plan(*pools[c],seed=seed*10000+c*100+rnd,
                                   steps=p['steps_per_client_per_round'][c])
                ds = tf.data.Dataset.from_tensor_slices((bx,by)).batch(p['batch_size'])
                opt = tf.data.Options(); opt.threading.private_threadpool_size = 1
                hist = model.fit(ds.with_options(opt),epochs=1,verbose=0)
                results.append((None,fitres(model.get_weights(),p['real_client_counts'][c])))
                rows.append(dict(round=rnd,client_id=c,loss=float(hist.history['loss'][0]),
                     steps=p['steps_per_client_per_round'][c],real_count=p['real_client_counts'][c],
                     pool_count=len(pools[c][1]),label_schedule_sha256=hashlib.sha256(by.tobytes()).hexdigest()))
            params,_ = strategy.aggregate_fit(rnd,results,[])
            weights = parameters_to_ndarrays(params); models[0].set_weights(weights)
            if not all(np.isfinite(w).all() for w in weights):
                raise ValueError('Nonfinite DNN weights')
            print(f'{arm} seed={seed} round={rnd}/5 elapsed={time.monotonic()-started:.1f}s',flush=True)
        models[0].save_weights(checkpoint)
        csv_write(out/'training.csv',list(rows[0]),rows)
        manifest.update(status='completed',completed_rounds=p['rounds'],client_failures=0,
                        seconds=time.monotonic()-started,checkpoint_sha256=sha(checkpoint),exit_code=0)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),exit_code=1)
        raise
    finally:
        dump(out/'manifest.json',manifest)


def evaluate(evidence,local):
    import tensorflow as tf
    from benchmark.run import compile_dnn, initial, probabilities
    p = frozen(evidence,local); seal_path = evidence/'evaluation-seal.json'
    if seal_path.exists():
        raise ValueError('Evaluation already completed; preserve the sealed result')
    runs = []
    for seed in p['seeds']:
        for arm in p['arms']:
            out = evidence/arm/f'seed-{seed}'; m = load(out/'manifest.json')
            checkpoint = local/arm/f'seed-{seed}'/'final.h5'
            if m['status'] != 'completed' or sha(checkpoint) != m['checkpoint_sha256']:
                raise ValueError('All 15 final checkpoints must be frozen before evaluation')
            runs.append(dict(arm=arm,seed=seed,manifest_sha256=sha(out/'manifest.json'),checkpoint_sha256=sha(checkpoint)))
    dump(evidence/'training-seal.json',dict(frozen_utc=utc(),runs=runs,test_read=False))
    d = np.load(local/'development.npz')
    for r in runs:
        arm,seed = r['arm'],r['seed']; out = evidence/arm/f'seed-{seed}'
        tf.keras.backend.clear_session(); model = compile_dnn(seed,initial(local,seed))
        model.load_weights(local/arm/f'seed-{seed}'/'final.h5')
        scores = {}; prediction_rows = []
        for split,key in zip(SPLITS,['hard','mixed']):
            probs = probabilities(model,d[f'x_{key}']); truth = d[f'y_{key}']
            scores[split] = metrics(truth,probs)
            for rid,sub,y,pr in zip(d[f'{key}_ids'],d[f'{key}_subtypes'],truth,probs):
                prediction_rows.append(dict(split=split,record_id=rid,subtype=sub,y_true=int(y),y_pred=int(pr.argmax()),
                                            **{f'p{i}':float(v) for i,v in enumerate(pr)}))
        csv_write(out/'predictions.csv',list(prediction_rows[0]),prediction_rows)
        dump(out/'metrics.json',scores)
        print(f'EVAL {arm} seed={seed} challenge={scores[SPLITS[0]]["macro_f1"]:.4f} mixed={scores[SPLITS[1]]["macro_f1"]:.4f}',flush=True)
    dump(seal_path,dict(completed_utc=utc(),runs=runs,test_read=False,scope='exploratory development validation only'))


def report(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = load(evidence/'plan.json'); seal = load(evidence/'evaluation-seal.json')
    scores = {(r['arm'],r['seed']):load(evidence/r['arm']/f'seed-{r["seed"]}'/'metrics.json') for r in seal['runs']}
    table = []; per_class = []
    for split in SPLITS:
        for arm in ARMS:
            values = [scores[(arm,s)][split] for s in SEEDS]
            row = dict(split=split,arm=arm)
            for name in ['macro_f1','accuracy','balanced_accuracy','normal_false_positive_rate','binary_attack_recall']:
                vals = [v[name] for v in values]
                row[name+'_mean'] = statistics.mean(vals); row[name+'_std'] = statistics.stdev(vals)
            table.append(row)
            for seed in SEEDS:
                for label,name in enumerate(CLASSES):
                    m = scores[(arm,seed)][split]
                    per_class.append(dict(split=split,arm=arm,seed=seed,label=label,family=name,support=m['support'][label],
                                          precision=m['precision'][label],recall=m['recall'][label],f1=m['f1'][label]))
    csv_write(evidence/'summary.csv',list(table[0]),table)
    csv_write(evidence/'per-class.csv',list(per_class[0]),per_class)
    comparisons = {}; decision = {}
    for dose in DOSES:
        gan = f'balanced-ctgan-{dose}'; comparisons[dose] = {}; passed = True
        for control in ['balanced-real',f'balanced-ros-{dose}']:
            delta = {}
            for split in SPLITS:
                delta[split] = {}
                for metric in ['macro_f1','normal_false_positive_rate']:
                    v = [scores[(gan,s)][split][metric]-scores[(control,s)][split][metric] for s in SEEDS]
                    delta[split][metric] = dict(values=v,mean=statistics.mean(v),std=statistics.stdev(v),positive_seeds=sum(x>0 for x in v))
            primary = delta[SPLITS[0]]['macro_f1']; secondary = delta[SPLITS[1]]['macro_f1']
            ok = (primary['mean'] >= .01 and primary['positive_seeds'] >= 2 and secondary['mean'] >= -.01 and
                  delta[SPLITS[0]]['normal_false_positive_rate']['mean'] <= .01 and
                  delta[SPLITS[1]]['normal_false_positive_rate']['mean'] <= .01)
            delta['meets_predeclared_rule'] = ok; passed &= ok; comparisons[dose][control] = delta
        decision[dose] = bool(passed)
    dump(evidence/'comparison.json',dict(gan_minus_control=comparisons,meets_pilot_rule=decision,
         scope='validation only; neither treatment constitutes confirmed generalization or GAN benefit'))
    fig,axes = plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    labels = ['Real','ROS +25%','CTGAN +25%','ROS +100%','CTGAN +100%']
    for ax,split,title in zip(axes,SPLITS,['Held-out subtype challenge','Previously used mixed validation']):
        for i,arm in enumerate(ARMS):
            vals = [scores[(arm,s)][split]['macro_f1'] for s in SEEDS]
            ax.scatter([i-.08,i,i+.08],vals,s=32,color='#246a9c' if 'ctgan' not in arm else '#b85c19')
            ax.plot([i-.2,i+.2],[statistics.mean(vals)]*2,color='black',lw=2)
        ax.set_xticks(range(5),labels,rotation=20); ax.set_ylabel('Macro F1 (all five families)')
        ax.set_title(title); ax.grid(axis='y',alpha=.25)
    fig.suptitle('Conditional GAN pilot: development data only, three paired seeds')
    fig.savefig(evidence/'comparison.png',dpi=170); plt.close(fig)
    names = {'balanced-real':'Balanced real only','balanced-ros-025':'Balanced ROS +25%',
             'balanced-ctgan-025':'Balanced CTGAN +25%','balanced-ros-100':'Balanced ROS +100%',
             'balanced-ctgan-100':'Balanced CTGAN +100%'}
    lines = ['# Conditional GAN validation pilot','',
        '**Exploratory development results. KDDTest+ was not read or re-evaluated.**','',
        'The original GAN gained on ordinary validation without transferring that gain to the v1 test set. '
        'This pilot tests a shared conditional tabular GAN, a capped dose, and class-balanced batches. '
        'All controls use the same batches by family, initial weights, final-round checkpoint rule and update budget.','',
        '## Protocol and boundaries','',
        f'- Core real training: **{sum(p["training_class_counts"]):,}** rows; class counts {p["training_class_counts"]}.',
        f'- Entire `back`, `portsweep`, `warezclient`, `loadmodule` subtypes are excluded from the new training and scaler/GAN fits. '
        f'The challenge combines those held-out records with original validation normals: counts {p["challenge_class_counts"]}.',
        f'- Mixed validation: {sum(p["mixed_validation_class_counts"]):,} original v1 validation records. '
        'Its normal records also appear in the challenge; the two panels are not independent.',
        '- One scaler is refitted on core real training only. One CTGAN per seed jointly fits core Probe/R2L/U2R records, '
        'with family and binary features treated as discrete. Sampling is filtered by actual output family, projected to valid domains, '
        'and exact core/generated patterns rejected using training data only.',
        '- Synthetic doses are 25% and 100% of each client\'s real minority-family count (rounded up), with matched client-only ROS. '
        'Balanced batches already repeat rare real records in every control; augmentation must improve on that stronger baseline.',
        f'- Two serial in-process Flower FedAvg clients; 5 rounds; batch 128; {p["steps_per_client_per_round"]} updates/client/round; '
        f'**{5*sum(p["steps_per_client_per_round"]):,} updates/run**; three seeds; 15 DNN runs and 3 CTGAN fits.',
        '- Final checkpoints for all treatments were frozen before these scores. No evaluation-based filtering or checkpoint selection.',
        '- **These are not fresh independent test sets:** mixed validation was scored before; held-out subtype records were in v1 training. '
        'Subtype selection and this design follow inspection of prior results. Treat gains as hypotheses, not confirmed generalization.',
        f'- Only **{p["challenge_class_counts"][4]} U2R challenge records**. The altered class mix makes macro F1 incomparable with v1 test F1. '
        'Seed standard deviations describe optimization variation, not statistical confidence or dataset-sampling uncertainty.',
        f'- Full 41-feature groups are disjoint, but reduced 23-feature overlap is {p["reduced_pattern_overlap"]}. '
        'This is not an exclusively novel-input benchmark.',
        '- Raw acquisition provenance remains unrecorded. Pooled preparation provides no private-silo, secure aggregation or privacy guarantee.','',
        '## Measured results','']
    for split in SPLITS:
        lines += [f'### {split}','', '| Treatment | Macro F1, mean ± SD | Balanced accuracy | Normal FPR |',
                  '| --- | ---: | ---: | ---: |']
        for row in table:
            if row['split'] == split:
                lines.append(f'| {names[row["arm"]]} | {row["macro_f1_mean"]:.4f} ± {row["macro_f1_std"]:.4f} | '
                             f'{row["balanced_accuracy_mean"]:.4f} | {row["normal_false_positive_rate_mean"]:.4f} |')
        lines.append('')
    lines += ['## Paired GAN evidence','',
        '| Dose | Control | Challenge F1 difference, mean ± SD | Positive seeds | Mixed F1 difference | Passes pilot rule |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for dose,controls in comparisons.items():
        for control,v in controls.items():
            a = v[SPLITS[0]]['macro_f1']; b = v[SPLITS[1]]['macro_f1']
            lines.append(f'| {DOSES[dose]:.0%} | {names[control]} | {a["mean"]:+.4f} ± {a["std"]:.4f} | '
                         f'{a["positive_seeds"]}/3 | {b["mean"]:+.4f} | {v["meets_predeclared_rule"]} |')
    passing = [f'{DOSES[k]:.0%}' for k,v in decision.items() if v]
    finding = ('A candidate met the exploratory pilot rule at '+', '.join(passing)+'. This warrants independent confirmation, not a test-performance claim.'
               if passing else 'Neither dose met the predeclared pilot rule. This pilot does not establish a reliable GAN advantage.')
    lines += ['', '**Finding:** '+finding,'',
        'The rule required at least +0.01 mean challenge macro F1 versus both real-only and matched-dose ROS, positive differences '
        'in at least two seeds per control, no more than +0.01 mean normal FPR, and no more than 0.01 mixed-validation F1 loss. '
        'It is an exploratory decision rule, not a significance test. Both doses are reported; no hidden sweep.','',
        '![All paired validation results](comparison.png)','',
        '## Evidence and next decision','',
        '- [Locked plan](plan.json), [summary](summary.csv), [all family metrics](per-class.csv), '
        '[paired differences and decision](comparison.json), [independent verification](verification.json).',
        '- Each augmentation folder records training-fit IDs, losses, conditional-sampling rejections, projection counts and training-only '
        'feature-distribution diagnostics. Diagnostics check resemblance and validity; they do not establish detection benefit or privacy.',
        '- Run folders contain training losses, paired label-schedule hashes, environment versions, model/input hashes, family metrics '
        'and complete real-record probabilities. Initial/final weights and synthetic feature pools remain outside Git.',
        '- Confirmation requires a separately selected, provenance-verified real evaluation source, mapped to a locked taxonomy and comparable '
        'feature schema. Freeze the candidate before examining that data. Do not keep optimizing against the already scored v1 test set.',
        '- If no candidate improves the controls, retain real-only/oversampling as the supported choices and investigate more real rare-attack '
        'coverage or richer features. GAN samples cannot restore attack patterns absent from the underlying observations.','']
    (evidence/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(meets_pilot_rule=decision,summary=table),indent=2),flush=True)


def run(evidence,local):
    from benchmark.run import configure
    configure(); p = frozen(evidence,local)
    if (evidence/'evaluation-seal.json').exists():
        raise ValueError('This experiment was already evaluated; preserve it')
    (evidence/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for seed in p['seeds']:
        augment(evidence,local,seed)
        for arm in p['arms']:
            train(evidence,local,seed,arm)
    evaluate(evidence,local); report(evidence)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','run','report'])
    ap.add_argument('--evidence',required=True,type=Path); ap.add_argument('--local',type=Path)
    ap.add_argument('--raw-development',type=Path)
    ap.add_argument('--prior-evidence',type=Path,default=Path('results/benchmark/nsl-kdd-v1'))
    ap.add_argument('--prior-local',type=Path,default=Path('.benchmark-local/nsl-kdd-v1'))
    a = ap.parse_args()
    if a.action == 'report':
        report(a.evidence)
    elif a.local is None:
        ap.error('--local is required for prepare/run')
    elif a.action == 'prepare':
        if a.raw_development is None: ap.error('--raw-development is required for prepare')
        prepare(a.raw_development,a.prior_evidence,a.prior_local,a.evidence,a.local)
    else:
        run(a.evidence,a.local)


if __name__ == '__main__':
    main()
