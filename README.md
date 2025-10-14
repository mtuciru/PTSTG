# PTSTG — PatchTST + Adaptive Graph (Unified Repository)

A minimal, production-friendly repository for **PTSTG** — a patch-based temporal encoder (Transformer) combined with a **learned adaptive graph**. One CLI, consistent behavior across datasets, zero external tooling. Data loaders, metrics, logging, and the train/eval loop live in-repo.

---

## Highlights

- **Unified CLI** for train/val/test across datasets
- **Two data formats** supported out-of-the-box:
  - Pre-split `*.npz` with `X:(B,T,N,C)` and `Y:(B,H,N,C)`
  - Single `data.npy:(T,N[,C])` with automatic sliding windows
- **No adjacency required** — graph is learned (adaptive)
- **Per-horizon metrics** (MAE/MAPE/RMSE) + averaged score
- **Warmup + Cosine** LR schedule, checkpointing, simple logs
- **Smoke test** via synthetic data generator

> **Notation:** `B`=batch, `T`=input length, `H`=horizon, `N`=nodes, `C`=channels.

---

## Quick Start

```bash
# 0) Python 3.9+ recommended

# 1) Install deps
pip install -r requirements.txt

# 2) (Option A) Synthetic data for a quick smoke test
python scripts/make_synthetic.py --dataset demo
python train.py --dataset demo --data_dir ./data/demo --mode train --max_epochs 3
python train.py --dataset demo --data_dir ./data/demo --mode test

# 2) (Option B) Real data
# Place one of:
#   data/<dataset>/{train,val,test}.npz  # each with arrays X and Y
#   data/<dataset>/data.npy              # (T,N) or (T,N,C); windows built automatically

# 3) Train / Validate / Test
python train.py --dataset metr --data_dir ./data/metr --mode train
python train.py --dataset metr --data_dir ./data/metr --mode val
python train.py --dataset metr --data_dir ./data/metr --mode test
```

---

## Datasets

- **METR-LA**: `--dataset metr` (defaults to `N=207`, inferred from data if present)
- **PEMS-BAY**: `--dataset pems` (defaults to `N=325`, inferred from data if present)
- **Custom**: `--dataset custom` (nodes/channels inferred from data)

The model uses **adaptive adjacency**, so **no precomputed graph** is required.

---

## Data Layouts

### A) Pre-split `.npz` (recommended for reproducibility)

```
data/<dataset>/
  ├─ train.npz   # contains arrays: X:(B,T,N,C), Y:(B,H,N,C)
  ├─ val.npz
  └─ test.npz
```

**Minimal example to create `.npz` files:**
```python
import numpy as np

# X: (B, T, N, C), Y: (B, H, N, C)
np.savez_compressed("data/myset/train.npz", X=X_train, Y=Y_train)
np.savez_compressed("data/myset/val.npz",   X=X_val,   Y=Y_val)
np.savez_compressed("data/myset/test.npz",  X=X_test,  Y=Y_test)
```

### B) Raw time series `data.npy`

```
data/<dataset>/
  └─ data.npy  # ndarray of shape (T,N) or (T,N,C)
```

- Sliding windows constructed on-the-fly using `--seq_len` (input window) and `--horizon` (forecast horizon)
- Default temporal split: **70/10/20** (train/val/test)

---

## CLI Reference (Key Flags)

Run `python train.py -h` for the full list.

| Flag | Type | Default | Notes |
|---|---|---:|---|
| `--dataset` | str | `metr` | `metr`, `pems`, `custom` |
| `--data_dir` | str | `./data/<dataset>` | Root folder for data |
| `--mode` | str | `train` | `train`, `val`, `test` |
| `--device` | str | `cuda`/`cpu` | Auto-detected by default |
| `--seed` | int | `42` | Repro configuration |
| `--seq_len` | int | `12` | Input length `T` |
| `--horizon` | int | `12` | Forecast horizon `H` |
| `--input_dim` | int | `1` | Channels in input |
| `--output_dim` | int | `1` | Channels in output |
| `--batch_size` | int | `64` |  |
| `--max_epochs` | int | `50` | Early-stop via `--patience` |
| `--patience` | int | `10` | Epochs w/o val improvement |
| `--lrate` | float | `1e-3` | Adam LR |
| `--wdecay` | float | `1e-4` | Adam weight decay |
| `--dropout` | float | `0.1` |  |
| `--clip_grad_value` | float | `5.0` | Gradient clipping (norm) |
| `--num_workers` | int | `2` | DataLoader workers |
| `--d_model` | int | `128` | Transformer dim |
| `--n_heads` | int | `4` | Attention heads |
| `--n_layers` | int | `2` | Transformer layers |
| `--patch_len` | int | `4` | Patch length for encoder |
| `--stride` | int | `2` | Patch stride |
| `--graph_rank` | int | `8` | Rank for adaptive graph |
| `--graph_layers` | int | `1` | Graph blocks depth |

---

## Examples

**Train on METR-LA with custom windows**
```bash
python train.py \
  --dataset metr --data_dir ./data/metr \
  --seq_len 12 --horizon 12 \
  --batch_size 64 --max_epochs 50 --patience 10
```

**Train on PEMS-BAY with deeper model**
```bash
python train.py \
  --dataset pems --data_dir ./data/pems \
  --seq_len 12 --horizon 12 \
  --d_model 192 --n_heads 6 --n_layers 3 \
  --graph_layers 2 --graph_rank 16
```

**Train on a custom dataset using `data.npy`**
```bash
python train.py \
  --dataset custom --data_dir ./data/myset \
  --seq_len 24 --horizon 12 \
  --batch_size 32
```

---

## Outputs, Logging, and Checkpoints

- Artifacts live under:
  ```
  ./experiments/<model_name>/<dataset>/
  ```
- Best checkpoint is saved as:
  ```
  final_model_s{seed}.pt
  ```
- In `--mode test`, the engine loads the best checkpoint (if present) and prints **per-horizon** and **average** metrics (MAE/MAPE/RMSE) with masking.

---

## Model Overview (High Level)

- **Temporal encoder:** patchify along time → linear projection → Transformer encoder (GELU, pre-norm)
- **Graph module:** learned **adaptive adjacency** (low-rank param), optional stack of lightweight node-mixing blocks
- **Head:** per-node projection to `H × output_dim`

This setup removes the need for an external adjacency and generalizes across datasets without format-specific code.

---

## Repo Layout

```
ptstg-repo/
├── train.py
├── requirements.txt
├── scripts/
│   └── make_synthetic.py
└── src/ptstg/
    ├── __init__.py
    ├── args.py          # CLI flags
    ├── data.py          # unified loader (npz X/Y or raw data.npy + sliding windows)
    ├── engine.py        # train/val/test loop, checkpointing, logging
    ├── logging.py
    ├── metrics.py       # MAE/MAPE/RMSE with masks
    └── models/
        └── ptstg.py     # Patch-style temporal encoder + adaptive graph blocks
```

---

## Troubleshooting

- **Shape errors**:  
  Ensure `X:(B,T,N,C)` and `Y:(B,H,N,C)` for `.npz` mode. For `data.npy`, shapes must be `(T,N)` or `(T,N,C)`.
- **GPU OOM**:  
  Lower `--batch_size`, `--d_model`, or `--n_layers`; consider larger `--stride` / smaller `--patch_len`.
- **Slow dataloading**:  
  Increase `--num_workers`, confirm data is on a fast disk.
- **Flat metrics**:  
  Check scaling (handled automatically), verify `seq_len/horizon` are sensible for the dataset cadence.

---

## Requirements

- `torch>=2.1`
- `numpy>=1.24`
- `tqdm>=4.66`

Install with:
```bash
pip install -r requirements.txt
```

---
## Citation

If you use this pipeline in your research or production, please cite:
```
@Article{app151910468,
AUTHOR = {Mkrtchian, Grach and Gorodnichev, Mikhail},
TITLE = {Patch-Based Transformer–Graph Framework (PTSTG) for Traffic Forecasting in Transportation Systems},
JOURNAL = {Applied Sciences},
VOLUME = {15},
YEAR = {2025},
NUMBER = {19},
ARTICLE-NUMBER = {10468},
URL = {https://www.mdpi.com/2076-3417/15/19/10468},
ISSN = {2076-3417},
DOI = {10.3390/app151910468}
}


```

