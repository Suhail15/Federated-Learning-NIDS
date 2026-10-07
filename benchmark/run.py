"""Train paired controls, seal final checkpoints, then evaluate real holdout once."""
import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from model import build_dnn_model
from .common import (ARMS, SEEDS, FEATURES, CLASSES, sha, dump, load, csv_write, metrics)


def configure():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()


def code_hash():
    import hashlib
    h = hashlib.sha256()
    for p in sorted(list(Path('benchmark').glob('*.py')) + [Path('model.py')]):
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def protocol(evidence, local):
    p = load(evidence / 'protocol.yaml')
    for path, key in [(local/'development.npz', 'development_sha256'), (local/'test.npz', 'test_sha256'),
                      (evidence/'preprocessing.json', 'preprocessing_sha256')]:
        if sha(path) != p[key]:
            raise ValueError(f'Frozen input changed: {path.name}')
    return p


def compile_dnn(seed, weights):
    tf.keras.utils.set_random_seed(seed)
    m = build_dnn_model(); m.set_weights(weights)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy')
    return m


def dataset(x, y, seed, batch=128):
    d = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(y), seed=seed, reshuffle_each_iteration=True).repeat().batch(batch)
    options = tf.data.Options(); options.threading.private_threadpool_size = 1
    return d.with_options(options)


def probabilities(model, x):
    return np.concatenate([model(x[i:i+512], training=False).numpy() for i in range(0, len(x), 512)])


def initial(local, seed):
    path = local / f'initial-{seed}.npz'
    if not path.exists():
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        np.savez(path, *build_dnn_model().get_weights())
    with np.load(path) as z:
        return [z[f'arr_{i}'] for i in range(len(z.files))]


def build_gan():
    # Architectures from notebooks/data_aug.ipynb; fresh instances per class.
    generator = tf.keras.Sequential([tf.keras.layers.Input(shape=(100,))] +
        [layer for width in [128, 256, 512, 1024] for layer in
         [tf.keras.layers.Dense(width, use_bias=False), tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU()]] +
        [tf.keras.layers.Dense(23, activation='sigmoid')])
    discriminator = tf.keras.Sequential([tf.keras.layers.Input(shape=(23,))] +
        [layer for width in [512, 256, 128] for layer in
         [tf.keras.layers.Dense(width, use_bias=False), tf.keras.layers.LeakyReLU(), tf.keras.layers.Dropout(.3)]] +
        [tf.keras.layers.Dense(1, activation='sigmoid')])
    return generator, discriminator


def project_synthetic(x, prep):
    scale, offset = np.asarray(prep['scale']), np.asarray(prep['min'])
    raw = (x-offset) / scale
    projected = raw.copy()
    for i, name in enumerate(FEATURES):
        if name in ['land', 'logged_in']:
            projected[:, i] = (raw[:, i] >= .5).astype(float)
        elif name in ['duration', 'hot', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']:
            projected[:, i] = np.maximum(0, np.rint(raw[:, i]))
        else:
            projected[:, i] = np.clip(raw[:, i], 0, 1)
    if not np.isfinite(projected).all():
        raise ValueError('Nonfinite generated records')
    return (projected*scale+offset).astype('float32'), {
        'samples_projected': int(np.any(projected != raw, axis=1).sum()),
        'features_projected': {name: int(np.sum(projected[:, i] != raw[:, i])) for i, name in enumerate(FEATURES)},
        'rejected': 0}


def augment(evidence, local, seed):
    p = protocol(evidence, local)
    target = evidence / 'augmentation' / f'seed-{seed}'
    pool_path = local / f'augmentation-{seed}.npz'
    if (target/'manifest.json').exists():
        m=load(target/'manifest.json')
        if m['code_sha256'] != code_hash() or sha(pool_path) != m['pool_sha256']:
            raise ValueError('Cannot reuse augmentation produced by different code or data')
        return
    target.mkdir(parents=True, exist_ok=True)
    d = np.load(local/'development.npz')
    x, y, clients, ids = [d[k] for k in ['x_train','y_train','clients','train_ids']]
    prep=load(evidence/'preprocessing.json')
    counts=[np.bincount(y[clients==c],minlength=5) for c in range(2)]
    quotas=[[max(0,int(n[1]-n[label])) if label in [2,3,4] else 0 for label in range(5)] for n in counts]
    generated, losses, fit_rows, metadata = {}, [], [], []
    start=time.monotonic()
    for label in [2,3,4]:
        tf.keras.backend.clear_session(); gan_seed=seed*100+label
        tf.keras.utils.set_random_seed(gan_seed)
        generator, discriminator=build_gan()
        g_opt=tf.keras.optimizers.legacy.Adam(1e-4); d_opt=tf.keras.optimizers.legacy.Adam(1e-4)
        bce=tf.keras.losses.BinaryCrossentropy()
        @tf.function
        def step(real):
            noise=tf.random.normal([128,100])
            with tf.GradientTape() as gt, tf.GradientTape() as dt:
                fake=generator(noise,training=True)
                real_score=discriminator(real,training=True); fake_score=discriminator(fake,training=True)
                gl=bce(tf.ones_like(fake_score),fake_score)
                dl=bce(tf.ones_like(real_score),real_score)+bce(tf.zeros_like(fake_score),fake_score)
            g_opt.apply_gradients(zip(gt.gradient(gl,generator.trainable_variables),generator.trainable_variables))
            d_opt.apply_gradients(zip(dt.gradient(dl,discriminator.trainable_variables),discriminator.trainable_variables))
            return gl,dl
        class_x=x[y==label]
        batch=int(p['gan_batches'][str(label)])
        ds=tf.data.Dataset.from_tensor_slices(class_x).shuffle(len(class_x),seed=gan_seed).batch(batch)
        opt=tf.data.Options(); opt.threading.private_threadpool_size=1; ds=ds.with_options(opt)
        for epoch in range(1,p['gan_epochs']+1):
            values=[]
            for real in ds:
                gl,dl=step(real); values.append((float(gl.numpy()),float(dl.numpy())))
            avg=np.mean(values,axis=0)
            if not np.isfinite(avg).all(): raise ValueError('Nonfinite GAN loss')
            losses.append(dict(label=label,epoch=epoch,steps=len(values),generator_loss=avg[0],discriminator_loss=avg[1]))
            if epoch%10==0:
                print(f'GAN seed={seed} class={label} epoch={epoch}/{p["gan_epochs"]} losses={avg.tolist()}',flush=True)
        total=sum(q[label] for q in quotas)
        chunks=[generator(tf.random.normal([min(1024,total-i),100]),training=False).numpy() for i in range(0,total,1024)]
        synthetic,projection=project_synthetic(np.concatenate(chunks),prep)
        generated[f'class_{label}']=synthetic
        for rid in ids[y==label]: fit_rows.append(dict(label=label,record_id=rid,seed=gan_seed))
        weights_dir=local/'gan'/f'seed-{seed}'; weights_dir.mkdir(parents=True,exist_ok=True)
        gp=weights_dir/f'class-{label}-generator.h5'; dp=weights_dir/f'class-{label}-discriminator.h5'
        generator.save_weights(gp); discriminator.save_weights(dp)
        metadata.append(dict(label=label,seed=gan_seed,real_fit_count=len(class_x),generated_count=total,
                             independent_initialization=True,generator_sha256=sha(gp),discriminator_sha256=sha(dp),projection=projection))
    np.savez_compressed(pool_path,**generated)
    csv_write(target/'fit-ids.csv',['label','record_id','seed'],fit_rows)
    csv_write(target/'losses.csv',['label','epoch','steps','generator_loss','discriminator_loss'],losses)
    lineage=[]
    rng=np.random.default_rng(seed)
    for c in range(2):
        for label in [2,3,4]:
            candidates=ids[(clients==c)&(y==label)]
            for j,source in enumerate(rng.choice(candidates,quotas[c][label],replace=True)):
                lineage.append(dict(client_id=c,label=label,duplicate_id=f'ros:{seed}:{c}:{label}:{j}',source_record_id=source))
    csv_write(target/'ros-lineage.csv',['client_id','label','duplicate_id','source_record_id'],lineage)
    dump(target/'manifest.json',dict(seed=seed,code_sha256=code_hash(),protocol_sha256=sha(evidence/'protocol.yaml'),
         pool_sha256=sha(pool_path),fit_ids_sha256=sha(target/'fit-ids.csv'),quotas=quotas,classes=metadata,
         gan_seconds=time.monotonic()-start,status='completed',gan_preparation='pooled training only; no client privacy claim'))


def fitres(weights,n):
    return FitRes(status=Status(code=Code.OK,message=''),parameters=ndarrays_to_parameters(weights),num_examples=n,metrics={})


def train(evidence,local,seed,arm):
    p=protocol(evidence,local); out=evidence/arm/f'seed-{seed}'
    if (out/'manifest.json').exists():
        old=load(out/'manifest.json')
        if old.get('status')=='completed' and old['code_sha256']==code_hash(): return
        raise ValueError('Existing incomplete/different-code run; retain it and use a new experiment ID')
    out.mkdir(parents=True,exist_ok=True)
    d=np.load(local/'development.npz'); x,y,clients=[d[k] for k in ['x_train','y_train','clients']]
    xval,yval=d['x_val'],d['y_val']
    tf.keras.backend.clear_session(); weights=initial(local,seed)
    pools=[]; counts=[]
    aug=np.load(local/f'augmentation-{seed}.npz') if arm=='fedavg-gan' else None
    rng=np.random.default_rng(seed)
    for c in range(2):
        xc,yc=x[clients==c],y[clients==c]
        n=np.bincount(yc,minlength=5); counts.append(len(yc))
        extras_x,extras_y=[],[]
        for label in [2,3,4]:
            quantity=max(0,int(n[1]-n[label]))
            if arm=='fedavg-gan':
                offset=0 if c==0 else max(0,int(np.bincount(y[clients==0],minlength=5)[1]-np.bincount(y[clients==0],minlength=5)[label]))
                extra=aug[f'class_{label}'][offset:offset+quantity]
            elif arm=='fedavg-ros':
                extra=xc[yc==label][rng.choice(int(n[label]),quantity,replace=True)]
            else: continue
            extras_x.append(extra); extras_y.append(np.full(quantity,label,dtype=np.int64))
        if extras_x:
            xc=np.concatenate([xc]+extras_x); yc=np.concatenate([yc]+extras_y)
        pools.append((xc,yc))
    models=[compile_dnn(seed,weights) for _ in range(1 if arm=='centralized' else 2)]
    strategy=fl.server.strategy.FedAvg(initial_parameters=ndarrays_to_parameters(weights))
    checkpoint=local/arm/f'seed-{seed}'/'final.h5'; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError: commit=None
    manifest=dict(arm=arm,seed=seed,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  code_commit=commit,code_sha256=code_hash(),protocol_sha256=sha(evidence/'protocol.yaml'),
                  python=platform.python_version(),platform=platform.platform(),determinism=True,
                  tensorflow=tf.__version__,flower=fl.__version__,threads={'intra':4,'inter':2},
                  initial_sha256=sha(local/f'initial-{seed}.npz'),real_client_counts=counts,
                  pool_counts=[len(z[1]) for z in pools],aggregation='Flower FedAvg; real-count weights',
                  execution='serial in-process clients; no network transport',optimizer_state='local Adam slots persist across rounds',
                  checkpoint_selection='final round',test_used_in_training=False,client_failures=0,
                  command=' '.join(sys.argv),checkpoint_relative=str(checkpoint.relative_to(local)))
    dump(out/'manifest.json',manifest)
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    log_names=['train.txt'] if arm=='centralized' else ['server.txt','client-0.txt','client-1.txt']
    logs={name:(out/name).open('w') for name in log_names}
    rows=[]; start=time.monotonic()
    try:
        for rnd in range(1,p['rounds']+1):
            rstart=time.monotonic()
            if arm=='centralized':
                steps=sum(p['steps_per_client_per_round'])
                history=models[0].fit(dataset(x,y,seed+rnd),steps_per_epoch=steps,epochs=1,verbose=0)
                weights=models[0].get_weights()
                logs['train.txt'].write(f'block={rnd} steps={steps} loss={history.history["loss"][0]}\n'); logs['train.txt'].flush()
            else:
                results=[]
                for c,model in enumerate(models):
                    model.set_weights(weights)
                    xc,yc=pools[c]; steps=p['steps_per_client_per_round'][c]
                    hist=model.fit(dataset(xc,yc,seed+100*c+rnd),steps_per_epoch=steps,epochs=1,verbose=0)
                    results.append((None,fitres(model.get_weights(),counts[c])))
                    logs[f'client-{c}.txt'].write(f'round={rnd} real={counts[c]} pool={len(yc)} steps={steps} loss={hist.history["loss"][0]}\n'); logs[f'client-{c}.txt'].flush()
                parameters,_=strategy.aggregate_fit(rnd,results,[])
                weights=parameters_to_ndarrays(parameters)
                models[0].set_weights(weights)
                logs['server.txt'].write(f'round={rnd} successful_clients=2 failures=0 weights={counts}\n'); logs['server.txt'].flush()
            if not all(np.isfinite(w).all() for w in weights): raise ValueError('Nonfinite model weights')
            m=metrics(yval,probabilities(models[0],xval))
            rows.append(dict(round=rnd,split='real-validation',macro_f1=m['macro_f1'],accuracy=m['accuracy'],loss=m['loss'],
                             cumulative_updates=rnd*sum(p['steps_per_client_per_round']),seconds=time.monotonic()-rstart))
            print(f'{arm} seed={seed} round={rnd}/{p["rounds"]} val_macro_f1={m["macro_f1"]:.4f} seconds={rows[-1]["seconds"]:.1f}',flush=True)
        models[0].save_weights(checkpoint)
        manifest.update(status='completed',completed_rounds=p['rounds'],optimizer_updates=p['rounds']*sum(p['steps_per_client_per_round']),
                        presented_rows=128*p['rounds']*sum(p['steps_per_client_per_round']),dnn_seconds=time.monotonic()-start,
                        checkpoint_sha256=sha(checkpoint),exit_code=0)
        csv_write(out/'round-metrics.csv',list(rows[0]),rows)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),exit_code=1,completed_rounds=len(rows))
        raise
    finally:
        dump(out/'manifest.json',manifest)
        for f in logs.values(): f.close()


def seal(evidence,local):
    p=protocol(evidence,local); frozen=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=evidence/arm/f'seed-{seed}'/'manifest.json'; m=load(path)
            if m['status']!='completed' or m['code_sha256']!=code_hash() or m['protocol_sha256']!=sha(evidence/'protocol.yaml'):
                raise ValueError('Every predeclared run must finish under the frozen code/protocol before test evaluation')
            if sha(local/m['checkpoint_relative'])!=m['checkpoint_sha256']: raise ValueError('Checkpoint changed')
            frozen.append(dict(arm=arm,seed=seed,manifest_sha256=sha(path),checkpoint_sha256=m['checkpoint_sha256']))
    dump(evidence/'evaluation-seal.json',dict(protocol_sha256=sha(evidence/'protocol.yaml'),test_sha256=p['test_sha256'],
         code_sha256=code_hash(),frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=frozen,
         selection='all four arms and all three seeds; final checkpoints; no test-based selection'))


def verify_seal(evidence,local):
    seal=load(evidence/'evaluation-seal.json')
    if seal['protocol_sha256']!=sha(evidence/'protocol.yaml') or seal['code_sha256']!=code_hash() or seal['test_sha256']!=sha(local/'test.npz'):
        raise ValueError('Frozen evaluation inputs changed')
    for r in seal['runs']:
        path=evidence/r['arm']/f'seed-{r["seed"]}'/'manifest.json'; m=load(path)
        if sha(path)!=r['manifest_sha256'] or sha(local/m['checkpoint_relative'])!=r['checkpoint_sha256']:
            raise ValueError('Frozen training manifest/checkpoint changed')
    return seal


def evaluate(evidence,local):
    frozen=verify_seal(evidence,local)
    test=np.load(local/'test.npz'); dev=np.load(local/'development.npz')
    for run in frozen['runs']:
        out=evidence/run['arm']/f'seed-{run["seed"]}'
        if (out/'metrics.json').exists(): continue
        tf.keras.backend.clear_session(); model=build_dnn_model()
        manifest=load(out/'manifest.json'); model.load_weights(local/manifest['checkpoint_relative'])
        exports=[]
        for split,x,y,ids in [('real-validation',dev['x_val'],dev['y_val'],dev['val_ids']),('real-test',test['x'],test['y'],test['ids'])]:
            proba=probabilities(model,x)
            for rid,truth,pp in zip(ids,y,proba):
                exports.append(dict(record_id=rid,split=split,y_true=int(truth),y_pred=int(pp.argmax()),**{f'p{i}':float(pp[i]) for i in range(5)}))
            if split=='real-test': result=metrics(y,proba)
        result.update(seed=run['seed'],arm=run['arm'],split='real-test',test_sha256=frozen['test_sha256'],
                      checkpoint_sha256=manifest['checkpoint_sha256'],evaluation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        csv_write(out/'predictions.csv',['record_id','split','y_true','y_pred','p0','p1','p2','p3','p4'],exports)
        csv_write(out/'per-class.csv',['label','class','precision','recall','f1','support'],[
            dict(label=i,**{'class':CLASSES[i]},precision=result['precision'][i],recall=result['recall'][i],f1=result['f1'][i],support=result['support'][i]) for i in range(5)])
        cm=np.array(result['confusion_matrix']); normalized=cm/cm.sum(axis=1,keepdims=True)
        csv_write(out/'confusion-matrix.csv',['true_label','predicted_label','count','row_fraction'],[
            dict(true_label=i,predicted_label=j,count=int(cm[i,j]),row_fraction=float(normalized[i,j])) for i in range(5) for j in range(5)])
        dump(out/'metrics.json',result)
        print(f'TEST {run["arm"]} seed={run["seed"]} macro_f1={result["macro_f1"]:.4f}',flush=True)
    from .report import report
    report(evidence,local)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True); parser.add_argument('--local',required=True)
    parser.add_argument('--stage',choices=['all','augment','train','seal','evaluate'],default='all')
    parser.add_argument('--seed',type=int,choices=SEEDS); parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args(); evidence,local=Path(args.evidence),Path(args.local); configure()
    if args.stage=='all':
        for seed in SEEDS:
            initial(local,seed); augment(evidence,local,seed)
            for arm in ARMS: train(evidence,local,seed,arm)
        seal(evidence,local); evaluate(evidence,local)
    elif args.stage=='augment': augment(evidence,local,args.seed)
    elif args.stage=='train': train(evidence,local,args.seed,args.arm)
    elif args.stage=='seal': seal(evidence,local)
    else: evaluate(evidence,local)


if __name__=='__main__': main()
