# Retained artifacts

The full evidence tables are downloadable in `evidence-tables.tar.gz` and verified member-by-member in `archive-index.json`. Extract them into this experiment directory to restore their exact relative paths.

The author retains `.benchmark-local/nsl-kdd-v1/` on the benchmark machine. It contains `development.npz`, `test.npz`, `initial-{seed}.npz`, `augmentation-{seed}.npz`, class GAN generator/discriminator `.h5` weights, and `<arm>/seed-{seed}/final.h5` weights. Their exact hashes appear in `artifacts.sha256`. These files are excluded from Git and are not remotely downloadable. Reproduction from the recorded raw-file hashes and frozen source is the portable path; hashes alone are not a substitute for retaining the original files. Exact weight reproduction across hardware/software may differ despite deterministic execution on this machine.

Raw files remain in the local `nsl-kdd` source folder, outside this repository. Original download URL/date is unknown. The checksum-qualified acquisition limit remains part of every interpretation of these scores.
