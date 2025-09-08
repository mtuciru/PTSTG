import os
import torch

from src.ptstg.args import get_public_config
from src.ptstg.logging import get_logger
from src.ptstg.data import load_dataset, get_dataset_info
from src.ptstg.metrics import masked_mae
from src.ptstg.engine import BaseEngine
from src.ptstg.models.ptstg import PTSTG


def set_seed(seed: int):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def main():
    parser = get_public_config()
    args = parser.parse_args()

    log_dir = os.path.join('./experiments', args.model_name, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f'record_s{args.seed}.log')
    logger.info(args)

    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, node_num = get_dataset_info(args.dataset, args.data_dir, logger)
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = PTSTG(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        graph_rank=args.graph_rank,
        graph_layers=args.graph_layers,
        dropout=args.dropout
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_epochs = min(3, max(1, args.max_epochs // 10))
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs - warmup_epochs),
                               eta_min=max(args.lrate/100, 1e-6))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    engine = BaseEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed
    )

    if args.mode == "train":
        try:
            engine.load_model(log_dir)
            logger.info("[Resume] Loaded best checkpoint from %s", log_dir)
        except Exception as e:
            logger.info("[Resume] No checkpoint to load (%s). Training from scratch.", e)

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
