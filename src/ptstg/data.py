import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - torch.from_numpy(self.mean).to(data)) / torch.from_numpy(self.std).to(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * torch.from_numpy(self.std).to(data) + torch.from_numpy(self.mean).to(data)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # Ensure 4D: (B,T,N,C) and (B,H,N,C)
        def ensure_4d(a, is_y=False):
            if a.ndim == 3:
                a = a[..., None]  # add C
            if a.ndim != 4:
                raise ValueError(f"Expected {'Y' if is_y else 'X'} to be 4D, got {a.shape}")
            return a
        self.X = ensure_4d(X, is_y=False).astype('float32')
        self.Y = ensure_4d(Y, is_y=True).astype('float32')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, horizon: int):
        # data: (T,N) or (T,N,C)
        if data.ndim == 2:
            data = data[..., None]  # (T,N,1)
        if data.ndim != 3:
            raise ValueError(f"Expected data (T,N) or (T,N,C), got {data.shape}")
        self.data = data.astype('float32')
        self.T, self.N, self.C = self.data.shape
        self.seq_len = seq_len
        self.horizon = horizon
        self._len = max(0, self.T - self.seq_len - self.horizon + 1)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x = self.data[idx: idx+self.seq_len]                         # (T,N,C)
        y = self.data[idx+self.seq_len: idx+self.seq_len+self.horizon]
        return x, y


class _LoaderWrapper:
    """Adapter layer to match .shuffle(), .get_iterator(), .num_batch API used by the engine."""
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self.num_batch = len(loader)

    def shuffle(self):
        # No-op: DataLoader with shuffle=True already shuffles per epoch
        pass

    def get_iterator(self):
        for batch in self._loader:
            X, Y = batch
            yield X.numpy(), Y.numpy()


def _npz_read_pair(path_npz: str):
    arr = np.load(path_npz)
    if 'X' in arr and 'Y' in arr:
        return arr['X'], arr['Y']
    else:
        raise ValueError(f"{path_npz} must contain arrays 'X' and 'Y'")


def _prepare_from_raw(data_dir: str, split: str, seq_len: int, horizon: int):
    # Expect <data_dir>/data.npy. We'll split by 7:1:2 along time by default.
    data_path = os.path.join(data_dir, 'data.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw series not found: {data_path}")
    series = np.load(data_path)  # (T,N) or (T,N,C)
    if series.ndim == 2:
        series = series[..., 0:1]  # ensure channel dim
    T = series.shape[0]
    n_train = int(0.7 * T)
    n_val = int(0.1 * T)
    if split == 'train':
        segment = series[:n_train]
    elif split == 'val':
        segment = series[n_train:n_train+n_val]
    else:
        segment = series[n_train+n_val:]
    ds = SlidingWindowDataset(segment, seq_len, horizon)
    # Build tensors:
    X = np.stack([ds[i][0] for i in range(len(ds))], axis=0)  # (B,T,N,C)
    Y = np.stack([ds[i][1] for i in range(len(ds))], axis=0)  # (B,H,N,C)
    return X, Y


def load_dataset(data_dir: str, args, logger):
    # Prefer pre-split npz: <split>.npz with X/Y. Otherwise, build from data.npy with sliding window.
    paths = {s: os.path.join(data_dir, f"{s}.npz") for s in ['train', 'val', 'test']}
    has_npz = all(os.path.exists(p) for p in paths.values())

    if has_npz:
        Xtr, Ytr = _npz_read_pair(paths['train'])
        Xva, Yva = _npz_read_pair(paths['val'])
        Xte, Yte = _npz_read_pair(paths['test'])
    else:
        Xtr, Ytr = _prepare_from_raw(data_dir, 'train', args.seq_len, args.horizon)
        Xva, Yva = _prepare_from_raw(data_dir, 'val', args.seq_len, args.horizon)
        Xte, Yte = _prepare_from_raw(data_dir, 'test', args.seq_len, args.horizon)

    # scaler from train:
    mean = Xtr.mean(axis=(0, 1), keepdims=True)  # mean over B,T
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    scaler = StandardScaler(mean.squeeze(), std.squeeze())

    # z-norm
    def z(a): return (a - mean) / std
    Xtr_z, Ytr_z = z(Xtr), z(Ytr)
    Xva_z, Yva_z = z(Xva), z(Yva)
    Xte_z, Yte_z = z(Xte), z(Yte)

    # Torch loaders
    train_ds = ArrayDataset(Xtr_z, Ytr_z)
    val_ds = ArrayDataset(Xva_z, Yva_z)
    test_ds = ArrayDataset(Xte_z, Yte_z)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    dct = {
        'train_loader': _LoaderWrapper(train_loader),
        'val_loader': _LoaderWrapper(val_loader),
        'test_loader': _LoaderWrapper(test_loader),
    }

    return dct, scaler


def get_dataset_info(dataset: str, data_dir: str, logger):
    # Defaults for popular datasets; falls back to inspect data_dir.
    default_nodes = {'metr': 207, 'pems': 325}
    if data_dir is None:
        data_dir = os.path.join('./data', dataset)

    node_num = default_nodes.get(dataset, None)

    # Try to infer node count from files if present
    try:
        for split in ['train', 'val', 'test']:
            p = os.path.join(data_dir, f"{split}.npz")
            if os.path.exists(p):
                arr = np.load(p)
                X = arr['X']
                node_num = X.shape[2]
                break
        else:
            p = os.path.join(data_dir, 'data.npy')
            if os.path.exists(p):
                d = np.load(p)
                if d.ndim == 2:
                    node_num = d.shape[1]
                elif d.ndim == 3:
                    node_num = d.shape[1]
    except Exception as e:
        if logger: logger.info("Node inference failed: %s", e)

    if node_num is None:
        raise ValueError("Cannot infer number of nodes; specify a standard dataset (metr|pems) or provide data to infer.")

    return data_dir, node_num
