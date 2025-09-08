import os
import torch
import numpy as np
from tqdm import tqdm

from .metrics import masked_mape, masked_rmse, compute_all_metrics

class BaseEngine():
    def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer,
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed):
        super().__init__()
        self._device = device
        self.model = model.to(device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._logger.info('The number of parameters: %s', self.model.param_num())

    # --- utils ---
    def _to_device(self, tensors):
        import torch
        if isinstance(tensors, list):
            return [t.to(self._device) for t in tensors]
        return tensors.to(self._device)

    def _to_tensor(self, arr):
        import torch
        if isinstance(arr, list):
            return [torch.tensor(x, dtype=torch.float32) for x in arr]
        return torch.tensor(arr, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        def inv(t): return self._scaler.inverse_transform(t)
        if isinstance(tensors, list):
            return [inv(t) for t in tensors]
        return inv(tensors)

    # --- io ---
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(save_path, f'final_model_s{self._seed}.pt'))

    def load_model(self, save_path):
        self.model.load_state_dict(
            torch.load(os.path.join(save_path, f'final_model_s{self._seed}.pt'), map_location=self._device)
        )

    # --- training ---
    def train_batch(self, epoch=None):
        self.model.train()
        train_mae, train_mape, train_rmse = [], [], []

        self._dataloader['train_loader'].shuffle()
        iterator = self._dataloader['train_loader'].get_iterator()
        total_batches = getattr(self._dataloader['train_loader'], 'num_batch', None)

        pbar = tqdm(total=total_batches, desc=f"Train {epoch+1}", leave=False)

        for step, (X, label) in enumerate(iterator, start=1):
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            mask_value = torch.tensor(0, device=pred.device)
            if label.min() < 1:
                mask_value = label.min()

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_mae.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
            pbar.update(1)

        pbar.close()
        return np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse)

    def train(self):
        self._logger.info('Start training!')
        wait, min_val_mae = 0, np.inf

        epbar = tqdm(range(self._max_epochs), desc="Epochs", leave=True)
        for epoch in epbar:
            mtrain_mae, mtrain_mape, mtrain_rmse = self.train_batch(epoch)
            mvalid_mae, mvalid_mape, mvalid_rmse = self.evaluate('val')

            cur_lr = self._lrate if self._lr_scheduler is None else self._lr_scheduler.get_last_lr()[0]
            if self._lr_scheduler: self._lr_scheduler.step()

            self._logger.info(
                'Epoch %03d | Train MAE %.4f RMSE %.4f MAPE %.4f | '
                'Val MAE %.4f RMSE %.4f MAPE %.4f | LR %.2e',
                epoch+1, mtrain_mae, mtrain_rmse, mtrain_mape,
                mvalid_mae, mvalid_rmse, mvalid_mape, cur_lr
            )

            epbar.set_postfix(train_mae=f"{mtrain_mae:.4f}", val_mae=f"{mvalid_mae:.4f}")

            if mvalid_mae < min_val_mae:
                self.save_model(self._save_path)
                min_val_mae, wait = mvalid_mae, 0
            else:
                wait += 1
                if wait == self._patience:
                    break

        self.evaluate('test')

    def evaluate(self, mode):
        if mode == 'test':
            try:
                self.load_model(self._save_path)
            except Exception:
                pass
        self.model.eval()

        preds, labels = [], []
        with torch.no_grad():
            iterator = self._dataloader[mode + '_loader'].get_iterator()
            for X, label in iterator:
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
        mask_value = labels.min() if labels.min() < 1 else torch.tensor(0)

        if mode == 'val':
            return (self._loss_fn(preds, labels, mask_value).item(),
                    masked_mape(preds, labels, mask_value).item(),
                    masked_rmse(preds, labels, mask_value).item())

        elif mode == 'test':
            test_mae, test_mape, test_rmse = [], [], []
            H = preds.shape[1]
            for i in range(H):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                self._logger.info('Horizon %d | MAE %.4f RMSE %.4f MAPE %.4f',
                                  i+1, res[0], res[2], res[1])
                test_mae.append(res[0]); test_mape.append(res[1]); test_rmse.append(res[2])
            self._logger.info('Test AVG | MAE %.4f RMSE %.4f MAPE %.4f',
                              np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape))
            return np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)
