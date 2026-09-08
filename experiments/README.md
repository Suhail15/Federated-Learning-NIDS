# Historical capstone scripts

These scripts preserve the earlier experiment variants. Use the root `client.py` and `server.py` for the documented demonstration.

Publication cleanup replaced machine-specific dataset paths with `data/` paths and corrected `12i8` to `128` in `clientdnn.py`. Run historical scripts from the repository root if exploring them.

Known historical limitations include fixed client IDs, inconsistent validation splits, evaluation on training data in early clients, and an experimental `clientdnn3exp.py` server that treats serialized Flower parameters as arrays. These scripts are research history, not independently validated launch commands.

The `clientdnn_confustionpy.py` script is a centralized training/plotting experiment, despite its client-like filename.
