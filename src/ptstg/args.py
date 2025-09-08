import argparse
import torch

def get_public_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PTSTG unified runner")

    # Core data/exp args
    parser.add_argument('--dataset', type=str, default='metr', help='metr|pems|custom')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='ptstg')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])

    # I/O window
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)

    # Optimization
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=2)

    # PTSTG hyperparams
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--patch_len', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--graph_rank', type=int, default=8)
    parser.add_argument('--graph_layers', type=int, default=1)

    return parser
