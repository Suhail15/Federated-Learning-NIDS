# Run the controlled benchmark

This runner leaves historical demos intact. It prepares original real records, fits preprocessing/GANs only on training IDs, trains paired controls with identical DNN update budgets, and freezes all 12 final checkpoints before any test evaluation.

The two clients run serially in process; Flower's actual FedAvg strategy aggregates their models. This benchmark does not test RPC transport or private raw-data silos. GAN and scaler preparation uses the pooled real training subset.

Use Python 3.9–3.11 and install `requirements-benchmark.txt`. From the repository root:

```bash
python -m pip install -r requirements-benchmark.txt
python -m unittest discover -s tests -v
python -m benchmark.prepare --train /path/to/KDDTrain+.txt --test /path/to/KDDTest+.txt --evidence results/benchmark/my-run --local .benchmark-local/my-run
python -m benchmark.run --evidence results/benchmark/my-run --local .benchmark-local/my-run
```

Each preparation uses a new evidence directory. Inputs/protocol are immutable. Complete augmentation and training stages can be reused after interruption only when code hashes match; retain failed runs instead of silently replacing them.

The fixed design uses two clients, four arms, seeds 11/22/33, five rounds, class GANs trained for 50 epochs, and real-count aggregation weights. The number of steps is computed from the frozen augmentation quotas; the no-augmentation controls repeat real records to match update counts. Local Adam state persists across rounds.

`--stage augment --seed 11` or `--stage train --seed 11 --arm fedavg-real` runs one stage. `--stage seal` requires all 12 checkpoints; `--stage evaluate` checks the seal before opening test arrays. All evaluation uses the final global model, fixed argmax predictions, and the same real test records.

Retain `.benchmark-local/<run>/`: it contains raw-derived arrays, fresh synthetic pools, initial/final weights, and GAN checkpoints. It is ignored by Git. Public evidence contains hashes, source/split IDs, logs, predictions, metrics and figures. A fresh clone requires original raw files; their exact checksums and the recorded label taxonomy must match for direct reproduction.

See [the protocol](../docs/benchmark-protocol.md) for leakage audits, metrics, taxonomy/provenance limits, and the separation from historical smoke validation.
