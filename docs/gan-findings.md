# Can GAN augmentation work here?

**The pipeline works, but the tested GAN configurations do not improve detection.** Two bounded conditional-GAN studies completed 30 DNN runs and six GAN fits. Every GAN configuration lost to its matched-dose random-oversampling control in all three seeds on the subtype challenge. The existing v1 test benchmark remains unchanged. These additional measurements use development data only.

## What was tried and measured

Both studies use the same audited core records, two client assignments, original DNN, balanced class exposure, paired initial weights, final-round checkpoint rule and 4,050 classifier updates per run. They compare real-only training with ROS and CTGAN at **25% and 100% additional minority records**, rounded up per client. Every synthetic dose is matched by an ordinary-oversampling control.

The first study jointly fits a small CTGAN to the natural minority pool. The follow-up retains every unique minority record and bootstraps equal family counts inside GAN fitting, with exact source-row lineage. Its 34 epochs give 1,394 adversarial updates versus the first study's 1,400; this is a mechanical budget adjustment, not an epoch sweep.

Challenge macro F1 is **mean ± sample SD across seeds 11, 22, 33**:

| Classifier input | Natural minority GAN fit | Family-balanced GAN fit |
| --- | ---: | ---: |
| Balanced real only | 0.3921 ± 0.0215 | 0.3921 ± 0.0215 |
| ROS +25% | 0.3891 ± 0.0083 | 0.3891 ± 0.0083 |
| CTGAN +25% | 0.3630 ± 0.0071 | 0.3516 ± 0.0081 |
| ROS +100% | 0.3996 ± 0.0312 | 0.3996 ± 0.0312 |
| CTGAN +100% | 0.3024 ± 0.0565 | 0.3445 ± 0.0097 |

The controls reproduce exactly across these two runs; they are repeated controls, not six independent seeds. Both doses in both studies fail the locked screening rule. The family-balanced fit improves the larger-dose GAN relative to the first attempt, but it still loses to ROS and real-only training. Neither study supports a GAN-benefit claim. Normal false positives and all family metrics are retained in the reports; accuracy alone is not the decision metric.

[First study and exact evidence](../results/gan-research/conditional-v1/report.md) · [Targeted follow-up and exact evidence](../results/gan-research/conditional-balanced-v1/report.md) · [Reproduction and locked protocol](../gan_research/README.md)

## What explains the difficulty?

There are concrete clues, with different evidence boundaries:

- **The original dose was extreme.** The v1 U2R GAN generated 36,365 records from 41 real examples; R2L generated 35,610 from 796. Feature distributions drifted, and ordinary-validation gains did not transfer to the v1 test file. [Prior scores and training-only diagnostics](../results/gan-research/conditional-v1/motivation.json) record the measurements. These clues motivate capped augmentation; they do not prove why the original GAN failed.
- **A generated family label does not establish realistic class semantics.** In seed 11, filtering the natural CTGAN's conditional output found the requested R2L/U2R label in only 7.0%/10.2% of requests. Rebalancing increased those rates to 31.4%/48.2%. U2R's generated `logged_in` mean moved from 0.147 to 0.588, closer to the real core mean 0.912. That is improved generation quality, not improved detection. [Natural-fit diagnostics](../results/gan-research/conditional-v1/augmentation/seed-11/diagnostics.csv) and [balanced-fit diagnostics](../results/gan-research/conditional-balanced-v1/augmentation/seed-11/diagnostics.csv) preserve the comparisons.
- **Rare support remains tiny.** Holding out entire subtypes leaves 81 real R2L and 34 real U2R core records. The U2R challenge has seven records. GAN samples, better conditioning and more epochs cannot supply additional real observations of missing attack behavior.
- **The current feature selection omits potentially useful information.** The notebook drops fields that are globally sparse, including failed logins and root-shell indicators. The benchmark preserves that 23-feature choice. In core training, `num_failed_logins > 0` occurs in **40/81 R2L** versus **55/53,784 Normal** records; `root_shell > 0` occurs in **17/34 U2R** versus **113/53,784 Normal** records. These are descriptive class statistics, not model scores. A generator working on the reduced inputs cannot restore omitted information. [Exact feature audit](../results/gan-research/feature-audit.json) records ten omitted fields.

Weak conditioning, lost feature information and limited rare support are plausible explanations. Their separate causal contributions have not been measured.

## Smallest useful next comparison — not run

Restore `num_failed_logins`, `root_shell`, `num_file_creations`, `num_compromised`, `num_shells` and `num_access_files` from the hash-verified raw development source. Start with these six numeric fields rather than adding every possible feature. Preserve integer/binary domains and fit all preprocessing on real core training only. Keep the existing full-record exclusion and subtype/client boundaries; rebuild both GAN inputs and the DNN input layer for the expanded feature schema.

Run three arms on that expanded schema: **balanced real only, matched ROS +25%, family-conditioned GAN +25%**, with the same three seeds, final-round rule, label schedules and update budgets. Refitting every control is necessary: any benefit from restoring features must be separated from the additional value of GAN data. Record per-family recall/F1/support, normal FPR, complete probabilities, conditional-label rates, training-only class-feature drift, exact source/fit IDs and checkpoints. Lock generator settings before scoring, report every attempted setting, and require GAN to improve both controls under the same screening rule.

This is the strongest next hypothesis from the current code/data audit, not an established solution. A positive development result would still need independent confirmation on a separately selected, provenance-verified real evaluation source with a compatible schema and locked taxonomy. The already scored KDDTest+ file must not become a repeated tuning target. If GAN still loses, use the stronger real-data/ROS model and describe the GAN as a tested research hypothesis.

Reproduce the descriptive audit without reading the test file:

```bash
python -m scripts.audit_rare_features \
  --raw-development ../nsl-kdd/KDDTrain+.txt \
  --local .benchmark-local/conditional-v1 \
  --output results/gan-research/feature-audit.json
```

## What is verified, and what is not?

**Verified:** two completed studies; all 30 final classifier checkpoints; six conditional GAN fits; complete saved real-record probabilities; independent metric and paired-comparison recomputation; training-only fit IDs; client-only ROS lineage; balanced-fit bootstrap lineage; matched classifier updates and paired family schedules. Automated tests cover the new fit boundary, full-group subtype isolation, capped quotas, domain projection and balanced fit/exposure.

**Not established:** consistent GAN improvement, benefit from the proposed expanded features, fresh independent generalization, privacy or memorization safety, realistic synthetic attack semantics, network performance or private-silo GAN training. The raw files' acquisition origin is still unrecorded. The challenge has an altered class mix and reduced-feature pattern overlap. Its records were in v1 training; mixed validation was previously scored, and normal records appear in both panels. Both studies are exploratory; three-seed SD is not statistical confidence.

CTGAN's mixed-type transforms and conditional sampling come from the [NeurIPS 2019 paper](https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf) and [pinned implementation](https://github.com/sdv-dev/CTGAN/tree/v0.11.0). Their suitability for tabular data motivates the experiment; it does not establish performance on this intrusion task.
