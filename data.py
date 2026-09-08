"""Validate the capstone CSV and form disjoint, repeatable client partitions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_partition(filename, client_id, num_clients, seed=42):
    if num_clients < 2 or not 0 <= client_id < num_clients:
        raise ValueError("Use at least two clients and a client ID from 0 to num_clients - 1.")
    frame = pd.read_csv(filename)
    frame.columns = frame.columns.str.strip()
    if len(frame.columns) != 24 or frame.columns[-1] != "class":
        raise ValueError("Expected 23 numeric feature columns followed by a 'class' column.")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("CSV contains missing or non-finite values.")
    labels = values[:, -1]
    if not np.isin(labels, np.arange(5)).all():
        raise ValueError("Class labels must be integers from 0 through 4.")
    if len(values) < num_clients * 5:
        raise ValueError("Need at least five rows per client for a train/validation split.")
    # Every process builds the same shuffled partition map. No rows are discarded.
    indices = np.random.default_rng(seed).permutation(len(values))
    partition = np.array_split(indices, num_clients)[client_id]
    train, val = train_test_split(partition, test_size=0.2, random_state=seed)
    return values[train, :-1], labels[train].astype(int), values[val, :-1], labels[val].astype(int)
