# Leakage-safe NSL-KDD benchmark results

Measured results from all four predeclared configurations and seeds 11, 22, 33. Each final checkpoint was frozen before any final test evaluation. No hyperparameter or checkpoint selection used test outcomes.

## Results

| Configuration | Macro F1 | Balanced accuracy | Accuracy | Normal false-positive rate |
| --- | ---: | ---: | ---: | ---: |
| centralized | 0.4658 ± 0.0038 | 0.4812 ± 0.0103 | 0.7472 ± 0.0092 | 0.0657 ± 0.0070 |
| fedavg-real | 0.4647 ± 0.0039 | 0.4788 ± 0.0018 | 0.7438 ± 0.0031 | 0.0582 ± 0.0112 |
| fedavg-ros | 0.4978 ± 0.0094 | 0.5097 ± 0.0023 | 0.7427 ± 0.0009 | 0.0823 ± 0.0038 |
| fedavg-gan | 0.4662 ± 0.0091 | 0.4768 ± 0.0090 | 0.7387 ± 0.0044 | 0.0688 ± 0.0023 |

Values are mean ± sample standard deviation across three paired training seeds. This is not a confidence interval or a significance test.

## GAN comparisons

- GAN minus fedavg-real: macro-F1 difference +0.0015 ± 0.0125; per-seed differences -0.0076, -0.0036, +0.0157.
- GAN minus fedavg-ros: macro-F1 difference -0.0316 ± 0.0002; per-seed differences -0.0315, -0.0314, -0.0319.

The oversampling comparison controls for class balancing and uses the same DNN update budget. GAN augmentation did not show consistent gains over real-only training; oversampling scored higher on macro F1 in every seed. Report the measured direction and variability; three seeds do not establish a universal GAN advantage.

## Data and execution boundaries

- Raw development file: 125,973 records. Final local KDDTest+ file: 22,544 real records, preserved in full.
- Removed 658 development records sharing full feature fingerprints with test records; quarantined 14 conflicting-label records in 7 groups. These exclusions may overlap; unique exclusion count is 664.
- Local raw-file hashes and expected names/counts are recorded. Their original acquisition source/date is unknown and independent official provenance is not verified. Scores refer to these exact locally held files and the explicit attack mapping.
- httptunnel is mapped to U2R; worm is mapped to DoS. Published taxonomies differ; test supports are recorded rather than assumed from other papers.
- Development is split before scaler/GAN fitting. GANs use pooled real training records only. Validation/test are real only; duplicate groups do not cross their full-feature boundaries.
- Two approximately IID clients, five rounds, batch size 128, client updates per round [779, 779]. All arms share total DNN update count and paired initial weights.
- Clients execute serially in process. Flower 1.8 FedAvg performs the actual model aggregation. This is not a new distributed network/transport test; the earlier network smoke test remains separate.
- Real-count aggregation weights, persistent local Adam state, unchanged capstone DNN, and final-round checkpoint selection are used. Pooled preprocessing/GAN preparation makes no private-silo claim.
- U2R has very small training support. GAN synthesis validity/memorization, non-IID clients, capstone-momentum, and real-network deployment remain unproven. There are 3,403 distinct 23-feature patterns shared between training and validation, and 2,789 shared between development and test, despite disjoint full original records. These projection collisions are retained and reported; this is not a benchmark on exclusively novel model-input patterns.

## Inspect the evidence

- [Frozen protocol](protocol.yaml), [provenance](provenance.json), [complete split/prediction tables](evidence-tables.tar.gz), [leakage audit](leakage-audit.json), [evaluation seal](evaluation-seal.json).
- [Every run](summary.csv), [paired comparisons](comparison.json), per-run manifests, predictions, per-class metrics, confusion matrices and validation curves.
- Initial/final model weights, generated pools, and raw/prepared datasets are retained in the ignored local artifact directory. The artifact hash index includes these local-only files; they are not bundled in Git.
- GAN manifests and fit-ID lists prove which real training records were used; projection counts and duplicate lineage are retained.

![Three-seed macro-F1 comparison](comparison.png)

## Full evidence tables

The [compressed table archive](evidence-tables.tar.gz) contains the exact split manifest, training-fit IDs/scaler parameters, GAN fit IDs, oversampling lineage, and all per-record predictions. The [archive index](archive-index.json) lists every member and hash. Extract it into this experiment directory to restore the original relative paths. Summary metrics and figures are directly viewable in GitHub. Raw features, generated feature pools, and checkpoints remain in the separately retained local artifact directory.

Independent verification recomputed metrics from all saved predictions and checked coverage, paired initial weights, update counts and augmentation lineage. See [the verification record](independent-verification.json).
