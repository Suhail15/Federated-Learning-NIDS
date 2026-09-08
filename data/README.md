# Data setup

Raw and generated datasets are intentionally kept outside version control. The repository contains the code and research notebooks; a fresh clone needs a prepared CSV before training.

## Existing capstone files

For the author's original folder layout, run this from the repository root:

```bash
cp "../Capstone Project/project_code/csv_files/df_aug_shuff_c1.csv" data/
```

Alternatively, pass the existing file directly without copying it:

```bash
python client.py --client-id 0 --num-clients 2 --data "../Capstone Project/project_code/csv_files/df_aug_shuff_c1.csv"
```

Use the identical path and settings for the other client, changing only its ID. The unaugmented `d_norm_train_t.csv` also meets the entry-point schema.

## Expected CSV

The header contains these 23 features, followed by `class`. Leading/trailing whitespace in column names is accepted.

```text
duration, land, hot, logged_in, count, srv_count,
serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
same_srv_rate, diff_srv_rate, srv_diff_host_rate,
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate,
dst_host_diff_srv_rate, dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate, dst_host_serror_rate,
dst_host_srv_serror_rate, dst_host_rerror_rate,
dst_host_srv_rerror_rate, class
```

All values must be finite and numeric. Classes are 0 (normal), 1 (DoS), 2 (Probe), 3 (R2L), and 4 (U2R). Prepared feature order must be consistent across all clients. Do not add an exported DataFrame index column.

## Raw NSL-KDD

Refer to the [UNB dataset page](https://www.unb.ca/cic/datasets/nsl.html) for provenance and access information. Existing local raw files live in the sibling `nsl-kdd` directory. For exploratory notebook use, copy the required `.txt` and `.arff` files into `data/raw/` and launch notebooks from `notebooks/`.

Raw files require feature selection, label mapping, and normalization before training. The exploratory notebooks do not fully reproduce every intermediate CSV from the historical project. Do not assume raw files alone reproduce the archived augmented dataset.

For new experiments, reserve real validation/test data first, fit transforms only on training records, and train augmentation models only on training records. Keep the untouched test set separate from client training data.
