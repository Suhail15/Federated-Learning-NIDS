# Portfolio evidence checklist

The [controlled benchmark](../results/benchmark/nsl-kdd-v1/report.md) is now complete, with measured tables, figures, full prediction evidence and independent verification. The metadata and demo-screenshot items below remain proposed additions. Prioritise inspectable evidence over star/fork counts. Keep the README's evidence table and leakage caveat when adding visuals.

## 1. Repository metadata

- **About description:** `NSL-KDD intrusion-detection capstone: TensorFlow DNN, Flower federated training, and GAN augmentation experiments. Leakage-audited benchmark with reproducible evidence and documented limits.`
- **Topics:** `federated-learning`, `intrusion-detection`, `network-security`, `nsl-kdd`, `flower`, `tensorflow`, `deep-learning`, `gan`, `python`, `capstone-project`.
- **Website:** leave empty until a working project page or demo exists; then link that exact page.
- **Social preview:** add `docs/images/social-preview.png`, 1280 × 640, showing the project title, stack, and architecture. Label it “Research prototype”; omit performance numbers.
- **Release:** after a fresh documented rerun, create `v0.1.0` titled “Local federated research prototype”. Include the tested commit, environment, commands, known leakage/data-reconstruction limitations, and evidence links. A release marks a reproducible code snapshot, not production readiness.
- **License:** choose a license only after confirming rights to publish the code; distinguish code licensing from NSL-KDD data terms. Do not add a license badge before that decision.
- **CI:** add a workflow for partition/input tests and syntax checks in the supported environment. Display its real status badge only once the workflow exists and has run; it does not verify benchmark performance.

## 2. Screenshots and figures

Place a compact “Demo and evidence” section immediately after the README evidence table once these files exist. Use relative image links and descriptive captions. Do not publish empty placeholders or simulated terminal output.

| Exact file | Capture or generate | Required caption / context |
| --- | --- | --- |
| `docs/images/federated-smoke-test.png` | Real server and both client terminals from a fresh one-round run, showing completion and failures | “Local workflow smoke test: two clients, one round, [actual row count], [actual failures]. Not benchmark evaluation.” Include run ID and link to full logs; redact private paths |
| `docs/images/class-distribution.png` | Bar chart from verified before/after counts, with readable minority classes (log scale or separate panels) | “Prepared dataset counts; synthetic rows increase class volume, not demonstrated detection quality.” Link to count CSV and provenance |
| `docs/images/architecture.png` | Export of the README architecture, if needed for the social preview | “One prepared CSV partitioned across local processes; parameter exchange, no secure aggregation or differential privacy.” Retain the Mermaid source |
| `docs/images/test-checks.png` | Actual test output from a fresh rerun | Record command, commit, date, environment, and pass count. Link to text log; this verifies data contracts only |

Keep benchmark confusion matrices and learning curves out of the main README until the leakage-safe evaluation below is complete. Historical plots, if recovered, must be labelled historical and potentially contaminated.

## 3. Inspectable result artifacts

Create `results/README.md` as an index explaining each run's scope and limitations. Use separate directories such as `results/smoke/<run-id>/` and `results/benchmark/<run-id>/` so smoke results cannot be confused with held-out evaluation.

For each fresh smoke run, save:

- `manifest.json`: run ID, UTC timestamp, commit SHA, OS/hardware, Python and dependency versions, exact commands, seeds, strategy, client count, rounds, epochs, batch size, actual row count, CSV SHA-256, sampling procedure, and split configuration.
- `server.log`, `client-0.log`, `client-1.log`, `tests.txt`: complete captured output and exit status; redact sensitive paths. Report actual failures and avoid reconstructing old logs.
- `class-counts.csv`: `label,class,before_count,augmented_count`, verified against source files. Explain file provenance and whether counts describe a full CSV or the sampled run.
- `summary.md`: what completed, what failed, and the explicit statement that augmented-before-split local validation does not establish generalization.

Do not commit raw data, checkpoints, or sensitive identifiers. Dataset hashes and acquisition/preparation instructions can document inputs without distributing them. No existing performance-export pipeline is implied by these filenames.

## 4. Tables worth adding

Retain the current component, class-count, and validation tables. Add these only with measured or recorded values:

| Table | Exact columns | Where it belongs |
| --- | --- | --- |
| Run configuration | Run ID, commit, environment, dataset hash, real/synthetic counts, split method, seed, strategy, clients, rounds, local epochs, batch size | `results/README.md`; link from README |
| Held-out performance | Experiment, seeds/run count, real test support, accuracy, macro F1, balanced accuracy, per-class precision/recall/F1, runtime, artifact link | README after a leakage-safe benchmark; use “not measured” until then |
| Controlled comparison | Centralized DNN / FedAvg / capstone-momentum, augmentation on/off, identical split and training budget, mean and variability across seeds | Benchmark report; do not declare a winner without comparable runs |

For per-class metrics, use a separate five-row table with class, real test support, precision, recall, and F1 so R2L/U2R behaviour is visible. Do not rely on overall accuracy for this imbalanced dataset.

## 5. Gate for credible benchmark evidence

1. Define and record the original real-record train/validation/test split before preprocessing or GAN training. State whether testing uses the official NSL-KDD test set or a custom held-out split.
2. Fit preprocessing and train the GAN using training records only. Partition training data across clients; keep validation/test data real and untouched by augmentation. Document partition assumptions and class coverage.
3. Compare centralized DNN, standard FedAvg, and capstone-momentum with and without augmentation under the same evaluation protocol. Record seeds and repeat runs; report variability and training budgets.
4. Save `metrics.json`, `per-class.csv`, `confusion-matrix.csv`, `confusion-matrix.png`, `round-metrics.csv`, and `learning-curves.png` with the run manifest. Record confusion-matrix label order and normalization; label local validation curves separately from held-out test metrics.
5. Link every README number or chart to its run artifacts. State whether changes exceed run-to-run variation. Keep unsuccessful or mixed results visible.

The highest-value immediate additions are the About/topics fields, a real smoke-test screenshot with saved logs, and a dataset-count chart with provenance. Benchmark plots are a later addition requiring a corrected experimental protocol.
