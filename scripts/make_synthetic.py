import os
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='demo')
    ap.add_argument('--T', type=int, default=2000)
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--C', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t = np.arange(args.T)
    base = np.sin(2*np.pi*t/48.0) + 0.3*np.sin(2*np.pi*t/7.0)
    data = np.stack([base + 0.1*rng.standard_normal(args.T) + 0.05*i for i in range(args.N)], axis=1)
    if args.C > 1:
        data = np.stack([data + 0.05*(j+1) for j in range(args.C)], axis=-1)  # (T,N,C)
    else:
        data = data[..., None]  # (T,N,1)

    root = os.path.join('data', args.dataset)
    os.makedirs(root, exist_ok=True)
    np.save(os.path.join(root, 'data.npy'), data.astype('float32'))
    print(f"Synthetic series saved to {root}/data.npy  shape={data.shape}")

if __name__ == "__main__":
    main()
