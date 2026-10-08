# glada (installable package)

The Python package behind every notebook in this repository. Notebooks only
configure runs and call into it; all model, training, data, and evaluation
logic lives here.

```bash
pip install -e .
```

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata and dependencies (`torch`, `transformers`, `soundfile`, `librosa`, `scikit-learn`, `scipy`, `numpy`, `tqdm`). |
| [`glada/`](glada) | The package source. See [`glada/README.md`](glada/README.md) for a file-by-file map. |
