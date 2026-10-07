"""Export real metrics, figures and complete evidence hashes."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from .common import ARMS, SEEDS, CLASSES, csv_write, dump, load, sha


def report(evidence,local):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data, summaries = {}, []
    for arm in ARMS:
        data[arm]=[]
        for seed in SEEDS:
            out=evidence/arm/f'seed-{seed}'; m=load(out/'metrics.json'); manifest=load(out/'manifest.json')
            data[arm].append(m)
            summaries.append(dict(arm=arm,seed=seed,macro_f1=m['macro_f1'],balanced_accuracy=m['balanced_accuracy'],
                                 accuracy=m['accuracy'],normal_false_positive_rate=m['normal_false_positive_rate'],
                                 binary_attack_recall=m['binary_attack_recall'],dnn_seconds=manifest['dnn_seconds'],status=manifest['status']))
            fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
            cm=np.array(m['confusion_matrix']); norm=cm/cm.sum(axis=1,keepdims=True)
            for ax,values,title in zip(axes,[cm,norm],['Real test: counts','Real test: row proportions']):
                ax.imshow(values,cmap='Blues',vmin=0,vmax=None if title.endswith('counts') else 1)
                ax.set_xticks(range(5),CLASSES,rotation=35,ha='right'); ax.set_yticks(range(5),CLASSES)
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
                for i in range(5):
                    for j in range(5): ax.text(j,i,str(cm[i,j]) if title.endswith('counts') else f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>values.max()*.55 else '#152b40')
            fig.suptitle(f'{arm} · seed {seed} · final global model'); fig.savefig(out/'confusion-matrix.png',dpi=150); plt.close(fig)
            with (out/'round-metrics.csv').open() as f: rows=list(csv.DictReader(f))
            fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
            axes[0].plot([int(r['round']) for r in rows],[float(r['macro_f1']) for r in rows],marker='o')
            axes[0].set_ylabel('Real validation macro F1'); axes[0].set_ylim(0,1)
            axes[1].plot([int(r['round']) for r in rows],[float(r['loss']) for r in rows],marker='o'); axes[1].set_ylabel('Real validation cross entropy')
            for ax in axes: ax.set_xlabel('Round / centralized block'); ax.grid(alpha=.2)
            fig.suptitle(f'{arm} · seed {seed} · validation only'); fig.savefig(out/'learning-curves.png',dpi=150); plt.close(fig)
    csv_write(evidence/'summary.csv',list(summaries[0]),summaries)
    summary={arm:{key:{'mean':float(np.mean([m[key] for m in data[arm]])),
                       'sample_sd':float(np.std([m[key] for m in data[arm]],ddof=1))}
                  for key in ['macro_f1','balanced_accuracy','accuracy','normal_false_positive_rate','binary_attack_recall']} for arm in ARMS}
    differences={}
    for control in ['fedavg-real','fedavg-ros']:
        delta=[d['macro_f1']-c['macro_f1'] for d,c in zip(data['fedavg-gan'],data[control])]
        differences[control]={'paired_deltas':delta,'mean':float(np.mean(delta)),'sample_sd':float(np.std(delta,ddof=1))}
    dump(evidence/'comparison.json',{'arms':summary,'gan_minus_control':differences,'seed_count':3,'interpretation':'descriptive paired variability; no significance claim'})
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    means=[summary[a]['macro_f1']['mean'] for a in ARMS]; sd=[summary[a]['macro_f1']['sample_sd'] for a in ARMS]
    bars=ax.bar(ARMS,means,yerr=sd,capsize=5,color=['#53779c','#406787','#57a49b','#d29a51'])
    ax.set_ylim(0,1); ax.set_ylabel('Real test macro F1'); ax.set_title('Final checkpoints · three paired seeds · mean ± sample SD')
    ax.set_xticks(range(4),['Centralized','FedAvg real','FedAvg oversampling','FedAvg GAN'])
    for bar,mean in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,mean+.045,f'{mean:.3f}',ha='center')
    fig.savefig(evidence/'comparison.png',dpi=160); plt.close(fig)
    provenance=load(evidence/'provenance.json'); audit=load(evidence/'leakage-audit.json'); protocol=load(evidence/'protocol.yaml')
    lines=['# Leakage-safe NSL-KDD benchmark results','',
           'Measured results from all four predeclared configurations and seeds 11, 22, 33. Each final checkpoint was frozen before any final test evaluation. No hyperparameter or checkpoint selection used test outcomes.','',
           '## Results','', '| Configuration | Macro F1 | Balanced accuracy | Accuracy | Normal false-positive rate |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        values=[f'{summary[arm][key]["mean"]:.4f} ± {summary[arm][key]["sample_sd"]:.4f}' for key in ['macro_f1','balanced_accuracy','accuracy','normal_false_positive_rate']]
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['', 'Values are mean ± sample standard deviation across three paired training seeds. This is not a confidence interval or a significance test.','', '## GAN comparisons','']
    for control,delta in differences.items():
        lines.append(f'- GAN minus {control}: macro-F1 difference {delta["mean"]:+.4f} ± {delta["sample_sd"]:.4f}; per-seed differences {", ".join(f"{x:+.4f}" for x in delta["paired_deltas"])}.')
    lines += ['', 'The oversampling comparison controls for class balancing and uses the same DNN update budget. Report the measured direction and variability; three seeds do not establish a universal GAN advantage.','',
              '## Data and execution boundaries','',
              f'- Raw development file: {provenance["train"]["rows"]:,} records. Final local KDDTest+ file: {provenance["test"]["rows"]:,} real records, preserved in full.',
              f'- Removed {audit["excluded_development_test_overlap"]} development records sharing full feature fingerprints with test records; quarantined {len(audit["conflicting_record_ids"])} conflicting-label records in {audit["conflicting_groups"]} groups. These exclusions may overlap; unique exclusion count is {provenance["excluded_development_records"]}.',
              '- Local raw-file hashes and expected names/counts are recorded. Their original acquisition source/date is unknown and independent official provenance is not verified. Scores refer to these exact locally held files and the explicit attack mapping.',
              '- httptunnel is mapped to U2R; worm is mapped to DoS. Published taxonomies differ; test supports are recorded rather than assumed from other papers.',
              '- Development is split before scaler/GAN fitting. GANs use pooled real training records only. Validation/test are real only; duplicate groups do not cross their full-feature boundaries.',
              f'- Two approximately IID clients, five rounds, batch size 128, client updates per round {protocol["steps_per_client_per_round"]}. All arms share total DNN update count and paired initial weights.',
              '- Clients execute serially in process. Flower 1.8 FedAvg performs the actual model aggregation. This is not a new distributed network/transport test; the earlier network smoke test remains separate.',
              '- Real-count aggregation weights, persistent local Adam state, unchanged capstone DNN, and final-round checkpoint selection are used. Pooled preprocessing/GAN preparation makes no private-silo claim.',
              '- U2R has very small training support. GAN synthesis validity/memorization, non-IID clients, capstone-momentum, and real-network deployment remain unproven. Reduced-feature collisions are reported separately from identical original records.','',
              '## Inspect the evidence','',
              '- [Frozen protocol](protocol.yaml), [provenance](provenance.json), [split manifest](splits.csv), [leakage audit](leakage-audit.json), [evaluation seal](evaluation-seal.json).',
              '- [Every run](summary.csv), [paired comparisons](comparison.json), per-run manifests, predictions, per-class metrics, confusion matrices and validation curves.',
              '- Initial/final model weights, generated pools, and raw/prepared datasets are retained in the ignored local artifact directory. The artifact hash index includes these local-only files; they are not bundled in Git.',
              '- GAN manifests and fit-ID lists prove which real training records were used; projection counts and duplicate lineage are retained.','',
              '![Three-seed macro-F1 comparison](comparison.png)','']
    (evidence/'report.md').write_text('\n'.join(lines))
    with (evidence/'artifacts.sha256').open('w') as f:
        for path in sorted(evidence.rglob('*')):
            if path.is_file() and path.name!='artifacts.sha256': f.write(f'{sha(path)}  public/{path.relative_to(evidence)}\n')
        for path in sorted(local.rglob('*')):
            if path.is_file(): f.write(f'{sha(path)}  local/{path.relative_to(local)}\n')
    print(json_summary(summary,differences),flush=True)


def json_summary(summary,differences):
    import json
    return json.dumps({'macro_f1':{a:summary[a]['macro_f1'] for a in ARMS},'gan_deltas':differences},indent=2)
