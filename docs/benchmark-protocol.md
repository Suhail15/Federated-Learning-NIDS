# Minimal leakage-safe benchmark protocol

Status: execution protocol. All four configurations and three paired seeds have completed. [Measured results and evidence](../results/benchmark/nsl-kdd-v1/report.md) document the execution. Prepared 7 October 2026 against public commit `d6705a356415b90b84cffd87c195e528674012cf`. The separate `benchmark` modules implement this protocol; historical demo entry points remain available.

## Question and scope

Does GAN augmentation improve five-class detection on untouched real NSL-KDD records relative to the same DNN trained with standard FedAvg, including a simple oversampling control?

The minimum experiment is one frozen data split, two simulated clients, four configurations, and three paired training seeds: **12 DNN fits and nine independent class-GAN fits**. This evaluates a local research workflow on an older benchmark. It does not establish real-network effectiveness, privacy, or performance across organizationally heterogeneous clients.

## Historical evidence and completed benchmark

| Item | Current status | Permitted interpretation |
| --- | --- | --- |
| Partition/input tests | README records 3/3 passed | Checks the current loader's row partitioning, repeatability, and invalid labels/client IDs; does not audit preprocessing or GAN fit provenance |
| Python compilation | README records success | Syntax validation only |
| Federated smoke test | Previously recorded: two clients, one round, 1,000 sampled augmented rows, zero client failures | Workflow completion only; augmented-before-split data prevents a generalization or GAN-benefit conclusion |
| Current code inspection | `data.py`, `client.py`, `server.py`, `model.py`, and augmentation notebook inspected for this protocol | Current loader splits its input CSV; clients report local accuracy; server provides FedAvg and the nonstandard capstone-momentum rule |
| Leakage-safe preprocessing, GAN training, and test evaluation | Implemented and run in the separate benchmark pipeline | Boundary audits and independent prediction verification passed |
| Benchmark scores and comparisons | All 12 final models evaluated on the sealed real test file | Report measured scope and mixed/negative GAN comparisons; historical smoke accuracy remains separate |

The historical network smoke test was not rerun; current test and compilation evidence is saved with the benchmark. Saved historical execution logs and benchmark artifacts are absent from the repository.

## 1. Freeze real data before learning anything from it

1. Use the complete raw `KDDTrain+.TXT` as the development source and the complete raw `KDDTest+.TXT` as the final test source. Do not use the 20% training subset, `KDDTest-21`, the 1,000-row smoke sample, or the historical prepared/augmented CSVs. The unaugmented prepared CSV is also unsuitable unless its transformation-fit provenance can be proved; rebuilding from raw records is the clean option.
2. Record acquisition provenance, exact file names, SHA-256 checksums, parser version, and raw row counts. UNB's current provenance page lists the files but says downloads are no longer available; use verifiable existing raw files or a documented mirror. File names alone do not prove official provenance.
3. Assign stable IDs from source-file hash plus original row number. Audit canonical full-feature fingerprints excluding labels and difficulty metadata. Keep identical real-feature records together across training/validation and client assignments. Quarantine all development records in full-feature groups with conflicting family labels; preserve test records and record every exclusion. This policy was fixed before training after the raw-data preflight found seven conflicting groups. Also report collisions after reducing to 23 features; reduced-feature equality is not itself proof of duplicated source records.
4. Freeze an explicit attack-name mapping to Normal/DoS/Probe/R2L/U2R, including test-only attack names within those five families. Reject unmapped labels. Use the 23 named features and their exact order from `data/README.md`; exclude the difficulty field. These are fixed schema decisions, not feature selection learned from validation/test outcomes.
5. Split only the original real training source into approximately **80% training / 20% validation**, stratified by five-class label with split seed **42**. Preserve any duplicate groups; record achieved fractions. All five classes must remain represented. Validation and final test contain real records only, retain their natural imbalance, and are never oversampled.
6. Audit train/validation/test overlap before fitting. For any exact full-feature overlap with the official test source, remove the matching development records under a predeclared rule, preserve the complete official test set, and record exclusions. Test-feature fingerprints are used only for this overlap audit; test outcomes remain sealed. Report any resulting modification to the official training source.
7. Fit one `MinMaxScaler` on the real training subset only, then apply that exact fitted transform to validation, test, and all client inputs. Do not fit another scaler on validation/test, recompute ranges after augmentation, or learn class targets from held-out counts. Keep real held-out values outside training ranges if they occur; do not use them to refit. This [split-before-fit rule](https://scikit-learn.org/stable/common_pitfalls.html) is the leakage boundary.

The test set is never used in client `fit`, GAN training, round evaluation, checkpoint selection, threshold selection, or hyperparameter adjustment. Unlock it for the final batch of evaluations only after configurations and checkpoints are frozen. Use argmax of the five softmax outputs; no test-tuned threshold.

## 2. Freeze client assignments before augmentation

- Use clients `0` and `1`, with deterministic, approximately equal, class-stratified partitions of real training IDs; partition seed **42**. Preserve duplicate groups. Every real training ID belongs to exactly one client and the two assignments cover the full training subset.
- Separately assign the real validation IDs to two disjoint client validation subsets, whose union is the frozen validation set. Do not recompute a 20% split of the augmented pools. Record class support; if a class is absent from a local validation subset, retain it in the pooled metrics and flag local metrics as unsupported.
- Reuse the exact real assignments, validation IDs, feature order, and preprocessing parameters in every arm and seed. Training seeds change initialization and sampling, not the split or client membership. This is an approximately IID simulation; non-IID and additional-client experiments are outside the minimum.
- Generate or duplicate rows only after these real assignments are frozen. Synthetic/duplicate rows are training-only, have unique IDs, and are assigned to exactly one client. Validation/test IDs and fitted-model inputs must remain disjoint by provenance; index disjointness alone is insufficient.
- Weight FedAvg aggregation by each client's fixed number of original real training records. Log augmented counts separately so synthetic volume cannot silently change client voting weight. Use identical weighting in all federated arms.

For the smallest experiment, use one **offline, pooled-training-only GAN per minority class**, then allocate its samples to the fixed client pools. This keeps the current single-dataset simulation scope and reduces GAN training overhead. The preprocessing fit and GAN preparation use pooled real training data, so this is not a demonstration of augmentation inside isolated private client silos. A fully decentralized follow-up would fit client-specific generators on each client's own training records.

## 3. Fresh augmentation with a matched control

For each training seed **11, 22, 33**, independently initialize the generator, discriminator, and both optimizers for each of Probe, R2L, and U2R. Fit only that class's real training records; never continue the same generator across classes as the historical notebook does. Use its existing architectures, latent dimension 100, Adam learning rate `1e-4`, and a predeclared 50 GAN epochs with class batches 512/32/2 respectively. Record exact implementation and random seeds; this is a new controlled implementation, not reproduction of the historical CSV.

For client `i`, let `n_i,c` be its original real training count for class `c` and let `t_i = n_i,DoS`. Add `max(0, t_i - n_i,c)` samples for each of classes 2–4; leave Normal and DoS unchanged. Derive these targets solely from the new training split, never the historical 45,927 count. The GAN sees the pooled real training subset for its class and receives no validation/test inputs. Allocate exactly the per-client quotas above.

Use the training-fitted scale compatible with the generator's sigmoid output. Freeze synthetic-domain handling in the schema: inverse-transform, enforce declared binary/integer/rate constraints by deterministic projection, and reapply the same scaler. Record rejection/projection counts and rules; do not choose them based on test scores. Save GAN fit-ID provenance, seeds, generated-pool hashes, and per-class real/synthetic counts. Tiny U2R training support can cause memorization or poor synthesis even after leakage is removed; this remains a limitation.

The oversampling control duplicates each client's own real minority training rows to exactly the same class counts as its GAN arm. Reuse the same real rows and client memberships. Retain source IDs for duplicates; do not split them into evaluation data.

## 4. Minimum comparisons and fixed DNN budget

| Arm | Training | Comparison it enables |
| --- | --- | --- |
| A | Centralized DNN, real training only | Reference for federated versus centralized training under this budget |
| B | Two-client standard FedAvg, real training only | Federated baseline |
| C | Two-client standard FedAvg, random oversampling | Controls for changing class balance through duplication |
| D | Two-client standard FedAvg, fresh GAN augmentation | Compared with B for augmentation-pipeline effect and C for GAN versus duplication |

Use `model.py` unchanged: 23 → 512 → 256 → 256 → 128 → 64 → 5 with its existing dropout. Keep Adam `1e-4`, batch size 128, loss, and preprocessing identical. Initialize all arms and clients from the same saved initial weights within each seed; do not rely on Flower choosing an arbitrary client's initialization.

Predeclare **five rounds**, both clients participating each round, and retain the final round's global checkpoint. Validation is diagnostic only; there is no early stopping, checkpoint search, or hyperparameter grid in this minimum.

Equal epochs would give augmented arms more optimizer updates. Instead, let `m_i` be client `i`'s total pool size after the augmentation target above, and set `S_i = ceil(m_i / 128)` DNN updates per round for **B, C, and D**. Shuffle/repeat each arm's own pool to deliver the same number of full batches; the unaugmented arm repeats its real data. Centralized A receives `5 × (S_0 + S_1)` updates on the union of real training rows, with validation recorded after each corresponding block. Log actual updates and row exposures. This matches total DNN update count, not the centralized/federated optimizer trajectory, which inherently differs. Report GAN cost separately.

Preserve and document the current client's Adam-state behaviour across rounds, use the same behaviour in B/C/D, and explicitly record TensorFlow determinism settings and hardware. Three paired seeds give descriptive variability, not a strong significance claim. If the predeclared budget underfits, report that result; extending the budget is a new protocol revision before further test evaluation.

`capstone-momentum` is outside this minimum because it introduces a second research question. To evaluate it, add real-only and GAN arms using the same protocol and compare them with B and D respectively. It is not standard FedAvgM. No aggregation-benefit claim is permitted until those comparisons run.

## 5. Held-out evaluation and reporting

Load each final global checkpoint in one evaluator and score every record in the same untouched real test set. Do not report a client's post-fit model as the global checkpoint. Also record pooled real-validation predictions from each round's global weights; keep them separate from final test results.

- **Primary:** five-class macro F1 with fixed label order `0,1,2,3,4`.
- **Secondary:** balanced accuracy (macro recall), overall accuracy, and per-class precision, recall, F1, and real support. Use `zero_division=0` and explicitly flag classes never predicted; require real test support for all five classes.
- **Operational summary:** normal false-positive rate = real normal records predicted as any attack / real normal support; binary attack recall = real attack records predicted as any attack / real attack support. These are distinct from correct attack-family recall.
- **Diagnostics:** 5×5 raw and row-normalized confusion matrices, real-validation loss/macro-F1 curves, DNN/GAN/total wall time, completed rounds/updates, and client failures.
- For each arm report each seed plus mean and sample standard deviation; report paired D−B and D−C macro-F1 differences for the same seeds. Keep failures/nonfinite runs visible, never replace them with favorable seeds. Separate stochastic-run variation from uncertainty about unseen real networks. Three seeds and a fixed test set do not justify a broad statistical or deployment claim.

## 6. Exact evidence artifacts

The following files are required outputs; completed outputs and their hashes are indexed with the measured benchmark.

Use `results/benchmark/<experiment-id>/` for the public evidence and a separate local artifact directory for raw/generated data and model weights.

| Path within experiment | Required content |
| --- | --- |
| `protocol.yaml` | Frozen split/partition/run seeds, four arms, targets, budgets, model/GAN settings, preprocessing/projection, aggregation weights, final checkpoint rule, metric definitions, protocol version |
| `provenance.json` | Raw source URLs/origins, retrieval date, names, hashes, row counts, label mapping version, exclusions; distinguish official provenance from a mirror claim |
| `schema.json`, `attack-map.json`, `preprocessing.json` | Ordered 23 features and domain types, complete five-family label map, training-fit IDs/hash and scaler parameters, synthetic projection policy |
| `splits.csv` | `record_id,source_file_sha256,source_row,feature_fingerprint,role,client_id`; IDs for real train/validation/test, no raw features |
| `class-counts.csv` | `stage,client_id,label,real_count,synthetic_count,duplicate_count`; explicit original, split, and post-augmentation counts |
| `leakage-audit.json`, `checks.txt` | Actual identity/group-overlap assertions, transform/GAN fit provenance, real-only evaluation checks, schema/domain checks, partition coverage/class support, test/compilation commands and exit codes |
| `<arm>/seed-<seed>/manifest.json` | UTC run ID/time, code commit and any patch, protocol/data/pool hashes, OS/hardware/Python/dependencies, exact commands, seeds, initial-weight hash, optimizer state policy, actual steps/rounds, failures, timing, checkpoint path/hash |
| `<arm>/seed-<seed>/environment.txt` | Exact installed package versions and determinism settings |
| `<arm>/seed-<seed>/server.txt`, `client-0.txt`, `client-1.txt` or `train.txt` | Actual full output and exit codes; centralized A uses `train.txt`; retain failed-run logs |
| `augmentation/seed-<seed>/manifest.json`, `fit-ids.csv`, `losses.csv` | Per-class GAN fit records/settings/seeds, generated hashes and allocation, ROS duplicate lineage, domain checks, independent initialization evidence, actual loss/step logs |
| `<arm>/seed-<seed>/predictions.csv` | `record_id,split,y_true,y_pred,p0,p1,p2,p3,p4` for final real evaluation; enough to recompute scores |
| `<arm>/seed-<seed>/metrics.json`, `per-class.csv` | All metric values/definitions, support, seed, split and checkpoint hashes; no placeholder numeric scores |
| `<arm>/seed-<seed>/confusion-matrix.csv`, `confusion-matrix.png` | Explicit label order, truth on rows, prediction on columns, counts and row-normalized views |
| `<arm>/seed-<seed>/round-metrics.csv`, `learning-curves.png` | Round-global real-validation metrics, actual optimizer updates, loss, timing; no repeatedly evaluated test curve |
| `summary.csv`, `report.md` | Every run, mean/sample SD, paired deltas, failures, evidence links, limits, and the existing no-benchmark claim until evaluation finishes |
| `artifacts.sha256` | Hashes of published evidence plus separately retained initial/final weights, generated pools, and fitted objects; include retrieval/storage instructions for local-only artifacts |

The current `.gitignore` excludes all CSVs, logs, NumPy archives, and model weights. A future evidence publication must narrowly allow safe result CSVs under `results/benchmark/` and text logs; do not broadly unignore datasets or checkpoints. Text/JSON evidence and PNGs can be published; retain raw records, synthetic pools, fitted objects, and initial/final checkpoints outside Git with hashes and recovery instructions. Hashes alone do not replace retained artifacts.

## 7. Implementation boundaries

- `benchmark.prepare` provides a raw-record parser and deterministic training-only preparation pipeline; the notebooks are not an end-to-end verified recipe.
- The benchmark uses explicit split/client manifests and separate training/real-validation inputs. `load_partition()` currently performs a non-stratified 20% split after reading the CSV, so `--data` pointing at another CSV does not implement this protocol.
- `benchmark.run` trains fresh class models/optimizers per seed with frozen fit IDs and matched oversampling.
- Common DNN update budgets, explicit initial parameters, real-count aggregation weights and final global checkpoint capture are implemented.
- The benchmark includes a held-out evaluator and artifact exports; current client/server metrics supply local accuracy, not this benchmark report. Select `--strategy fedavg` explicitly; the current default is capstone-momentum.
- Tests, boundary audits and independent verification check fit provenance, immutable membership, duplicate groups, real-only evaluation, scaler reuse, augmentation lineage and aggregation/update budgets. The existing three tests remain useful but do not verify these boundaries.

The completed report describes measured performance on this untouched NSL-KDD test set under this exact protocol. GAN benefit requires reporting the paired comparisons, including the oversampling control and rare-class support. Real deployment, privacy, robustness to non-IID clients, and capstone-momentum improvement remain separate, untested claims.

## Sources inspected

- [README at the reviewed commit](https://github.com/Suhail15/Federated-Learning-NIDS/blob/d6705a356415b90b84cffd87c195e528674012cf/README.md)
- [Current splitting implementation](https://github.com/Suhail15/Federated-Learning-NIDS/blob/d6705a356415b90b84cffd87c195e528674012cf/data.py)
- [Current client](https://github.com/Suhail15/Federated-Learning-NIDS/blob/d6705a356415b90b84cffd87c195e528674012cf/client.py) and [server](https://github.com/Suhail15/Federated-Learning-NIDS/blob/d6705a356415b90b84cffd87c195e528674012cf/server.py)
- [Historical GAN notebook](https://github.com/Suhail15/Federated-Learning-NIDS/blob/d6705a356415b90b84cffd87c195e528674012cf/notebooks/data_aug.ipynb)
- [UNB NSL-KDD provenance and train/test file descriptions](https://www.unb.ca/cic/datasets/nsl.html)
- [scikit-learn guidance on split-before-fit and shared transformations](https://scikit-learn.org/stable/common_pitfalls.html)

## Execution clarifications fixed before training

The benchmark uses serial clients in one process and calls Flower 1.8 FedAvg aggregation directly. It records this distinction from the earlier network smoke test. Original local-file acquisition is unrecorded, so the report qualifies provenance. The explicit label map assigns httptunnel to U2R and worm to DoS; it is frozen before training and test outcomes, and its exact supports are reported. Model/GAN weights and generated pools remain outside Git.

## Completed execution

The fixed run retained 100,247 real training and 25,062 real validation records, with all 22,544 local test records preserved. Development/test overlap excluded 658 records; seven conflicting groups contained 14 records, eight already in the overlap exclusion, for 664 unique exclusions. Nine tests and independent verification of all 12 runs passed. The full tables are packaged in `results/benchmark/nsl-kdd-v1/evidence-tables.tar.gz`; checkpoints, generated feature pools and prepared data remain in `.benchmark-local/nsl-kdd-v1/`. Source acquisition remains unverified. No additional configuration was selected after inspecting test results.
