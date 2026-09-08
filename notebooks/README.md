# Research notebooks

Install `requirements-notebooks.txt`, then launch Jupyter from this directory:

```bash
python -m pip install -r requirements-notebooks.txt
cd notebooks
jupyter lab
```

`clientmk3i.ipynb` contains exploratory preprocessing; `data_aug.ipynb` and `data_aug_exp.ipynb` contain GAN experiments; `clientdnn_confustion.ipynb` trains a centralized DNN and plots its confusion matrix and learning curves.

Outputs and execution counters were cleared for publication. Machine-specific paths were replaced with `../data/` and `../data/raw/`. Other relative paths and intermediate-file assumptions remain part of the historical experiments; review them before running cells.

The preprocessing notebook mixes binary-label and multiclass experiments, fits separate scalers in some exploratory cells, and contains optional/commented exports. It is not a verified automatic reconstruction of `df_aug_shuff_c1.csv`. Keep a consistent training-fitted transformation and untouched real test set when designing new evaluations.
