# Conditional GAN research pilot

The evaluated [v1 benchmark](../results/benchmark/nsl-kdd-v1/report.md) found no consistent GAN advantage. The original class GANs improved ordinary validation but did not transfer that gain to the real test file. This directory tests a specific alternative without reopening that test set.

**Research question:** does a shared conditional tabular GAN add useful minority-family variation beyond class-balanced real training and the same dose of random oversampling?

## Locked, bounded design

1. Reuse v1's audited original-record development split and client assignments. Exclude entire `back`, `portsweep`, `warezclient`, and `loadmodule` attack subtypes from the new training pool, keeping all identical full-feature groups together. The selection is deliberate, not random: one subtype per attack family, with enough remaining real support to train each family. It leaves very little rare-attack data.
2. Refit the scaler on the remaining **real core training only**. Re-read the hash-verified original development source instead of reversing the old float32-scaled array. The prior test-based quarantine remains inherited; do not read `KDDTest+` or `test.npz` again.
3. Fit one [CTGAN](https://github.com/sdv-dev/CTGAN/tree/v0.11.0) per seed jointly to real core Probe/R2L/U2R records. Family, `land`, and `logged_in` are discrete. Use mode-aware tabular transforms, conditional sampling and small networks: embedding 32, generator/discriminator `(128,128)`, batch 500, PacGAN 10, 100 epochs, CPU, library default learning rates/decays. All transforms fit inside the core pool.
4. Check the **actual generated family**, since the library's condition increases its probability rather than guaranteeing it. Reject wrong labels and nonfinite records; round nonnegative count features and two-decimal rates; enforce binary values and empirical within-family constants from core data only. Reject exact matches to any real core pattern and repeated accepted synthetic patterns. Cap sampling attempts; fail visibly if quotas cannot be supplied. These checks establish domain validity, not usefulness or privacy.
5. Add either **25% or 100% of each client's own real minority-family count**, rounded up. Use the same quotas for client-only random oversampling. The 25% pools are prefixes of the 100% pools. No DoS-sized balancing target and no substitute oversampling if GAN generation fails.
6. Compare five arms: balanced real, ROS +25%, CTGAN +25%, ROS +100%, CTGAN +100%. Every arm uses uniform family exposure in its mini-batches, identical family schedules, the original DNN, Adam `1e-4`, two serial Flower FedAvg clients weighted by real counts, five rounds and the same update budget. Steps are calculated once from the largest pool. Local Adam slots persist across rounds; initial weights are paired per seed. Run all three seeds `11,22,33` and retain every result.
7. Freeze **all 15 final checkpoints before scoring**. Primary: withheld-subtype records plus original validation normals. Secondary: the entire original mixed validation. Record accuracy, macro F1 including all five families, balanced accuracy, family precision/recall/F1/support, confusion matrices, normal false-positive rate and attack recall, with complete per-record probabilities.

The decision rule is declared in `plan.json` before fitting: a candidate needs at least **+0.01 mean challenge macro F1 versus both balanced real and matched-dose ROS**, positive paired differences in at least two of three seeds for each control, no more than +0.01 mean normal FPR, and no more than 0.01 mean secondary F1 loss versus either control. Report both doses, whether they pass or fail. This rule is an exploratory screening criterion, not a significance test.

## Interpretation boundary

This design follows inspection of v1's test and validation results. The mixed validation has already been evaluated, and the subtype challenge contains records used in v1 training. Both are held out **from the new fits**, but neither is an independent final test. Normal records are reused across both panels. The subtype challenge has a deliberately changed class mixture, and U2R support is extremely small. Seed standard deviations do not quantify dataset uncertainty. Full-record groups are disjoint; the reduced 23 features can still share identical patterns, and their overlap is recorded.

The new sampler and budget differ from v1, so compare arms **within this pilot**, not directly against v1's test scores. A promising result requires confirmation on a separately selected, provenance-verified real source with a locked feature schema and taxonomy. Do not repeatedly tune against the already evaluated v1 test file. A failure is also useful evidence: more synthetic rows cannot supply absent attack patterns or replace more real rare-attack coverage.

The pooled GAN and scaler do not implement private-silo federated preparation, differential privacy or secure aggregation. Exact-match rejection alone does not measure memorization or privacy risk. Raw-source acquisition provenance remains unrecorded.

## Reproduce

Use the existing Python 3.9 benchmark environment, or install the pinned optional dependencies in a separate environment. Tested versions are recorded with results; the package constraints preserve the original TensorFlow/Flower stack.

```bash
python -m pip install -r requirements-gan.txt
python -m gan_research.pilot prepare \
  --raw-development ../nsl-kdd/KDDTrain+.txt \
  --prior-evidence results/benchmark/nsl-kdd-v1 \
  --prior-local .benchmark-local/nsl-kdd-v1 \
  --evidence results/gan-research/conditional-v1 \
  --local .benchmark-local/conditional-v1
python -m gan_research.pilot run \
  --evidence results/gan-research/conditional-v1 \
  --local .benchmark-local/conditional-v1
python scripts/verify_gan_pilot.py --evidence results/gan-research/conditional-v1
python scripts/package_gan_pilot.py --evidence results/gan-research/conditional-v1
```

The prior local development arrays are produced by [v1 raw preparation](../benchmark/README.md). They preserve the exact audited IDs and client assignments; they are not shipped in Git. For an already populated evidence directory, extract `evidence-tables.tar.gz` first if verification needs its large tables. The run refuses to overwrite an evaluated pilot. Resuming completed training is allowed only with matching code and inputs.

Published evidence includes a locked plan, split/fit IDs, ROS lineage, CTGAN losses, conditional-sampling and projection audits, training-only distribution diagnostics, model/input/code hashes, client losses and label schedules, all family metrics, per-record probabilities, paired comparisons, an independent verification and a chart. Large ID/prediction tables are compressed; raw feature records, model weights, generated arrays and pickled CTGAN objects remain outside Git.

## One targeted follow-up

The first pilot did not improve the matched controls. Its training-only audits showed that Probe dominated the shared fit and rare-family conditional outputs poorly matched real rare-family features. The follow-up enables `prepare --balance-gan-fit` and uses experiment/output directories `conditional-balanced-v1`. Every unique core minority record is retained, then additional real copies are bootstrapped to equal family counts **inside GAN fitting only**. Exact source-row lineage is retained. This reweights existing evidence; it does not create more real observations.

The same 7,025 unique real records give 20,730 GAN fit rows. Epochs are mechanically reduced from 100 to 34: 1,394 adversarial updates versus the first pilot's 1,400. DNN controls, split boundaries, output doses, final-checkpoint rule and decision thresholds stay the same. All three seeds and both doses are retained. The design follows the first pilot's results and diagnostics, so its findings remain exploratory. No further tuning is hidden.

The modeling rationale comes from the [CTGAN paper, NeurIPS 2019](https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf): mixed data types, multimodal continuous columns and discrete imbalance motivate conditional tabular generation. That rationale does not establish an advantage on this intrusion task.
