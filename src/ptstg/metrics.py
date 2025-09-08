import torch

def _make_mask(y_true, mask_value):
    if isinstance(mask_value, torch.Tensor):
        mv = mask_value.to(y_true.device)
    else:
        mv = torch.tensor(mask_value, device=y_true.device, dtype=y_true.dtype)
    mask = (y_true != mv).float()
    return mask

def masked_mae(y_pred, y_true, mask_value=0):
    mask = _make_mask(y_true, mask_value)
    diff = (y_pred - y_true).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def masked_mape(y_pred, y_true, mask_value=0, eps=1e-5):
    mask = _make_mask(y_true, mask_value)
    denom = (y_true.abs() + eps)
    diff = ((y_pred - y_true).abs() / denom) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff.sum() / denom) * 100.0

def masked_rmse(y_pred, y_true, mask_value=0):
    mask = _make_mask(y_true, mask_value)
    diff2 = ((y_pred - y_true) ** 2) * mask
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt(diff2.sum() / denom)

def compute_all_metrics(y_pred, y_true, mask_value=0):
    return (
        float(masked_mae(y_pred, y_true, mask_value).item()),
        float(masked_mape(y_pred, y_true, mask_value).item()),
        float(masked_rmse(y_pred, y_true, mask_value).item()),
    )
