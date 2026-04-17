"""
Experiment class for time series imputation.

Benchmark protocol (matching TimeMixer++ Table 4):
  - seq_len = 1024
  - Randomly mask {12.5%, 25%, 37.5%, 50%} of time points
  - Loss on masked positions only (observed positions are free)
  - Final result averaged across the 4 mask ratios
  - Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather
  - Metric: MSE and MAE on masked positions

Masking strategy: random uniform masking, independent per sample and
per channel. This is the standard approach in the literature.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchLinear
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchLinear': PatchLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _mask_input(self, x):
        """
        Randomly mask mask_rate fraction of timesteps.
        Returns:
            x_masked : [B, L, C]  — zeros at masked positions
            mask     : [B, L, C]  — 1 = observed, 0 = masked
        """
        mask = torch.rand_like(x) > self.args.mask_rate   # True = keep
        mask = mask.float()
        x_masked = x * mask
        return x_masked, mask

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                x_masked, mask = self._mask_input(batch_x)
                mask = mask.to(self.device)

                outputs = self.model(x_masked, batch_x_mark, None, None, mask)

                # Loss on masked positions only
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                total_loss.append(loss.item())
        self.model.train()
        return np.mean(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data,  vali_loader  = self._get_data('val')
        test_data,  test_loader  = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        criterion      = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss  = []
            iter_count  = 0
            time_now    = time.time()
            epoch_start = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in \
                    enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                x_masked, mask = self._mask_input(batch_x)
                mask = mask.to(self.device)

                outputs = self.model(x_masked, batch_x_mark, None, None, mask)

                # Loss only on masked (held-out) positions
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | "
                        f"loss: {loss.item():.6f} | "
                        f"speed: {speed:.4f}s/iter | left: {left_time:.1f}s"
                    )
                    iter_count = 0
                    time_now   = time.time()

            train_loss = np.mean(train_loss)
            vali_loss  = self.vali(vali_loader, criterion)
            test_loss  = self.vali(test_loader,  criterion)

            print(
                f"Epoch {epoch+1} | "
                f"time: {time.time()-epoch_start:.1f}s | "
                f"train: {train_loss:.6f} | "
                f"vali: {vali_loss:.6f} | "
                f"test: {test_loss:.6f}"
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')
        if test:
            ckpt = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt))

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                x_masked, mask = self._mask_input(batch_x)
                mask = mask.to(self.device)

                outputs = self.model(x_masked, batch_x_mark, None, None, mask)

                # Collect only masked positions for evaluation
                outputs_np = outputs.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                mask_np    = mask.detach().cpu().numpy()

                # Flatten to [N_masked, ] for metric computation
                preds.append(outputs_np[mask_np == 0])
                trues.append(batch_x_np[mask_np == 0])

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print(f'imputation  mse: {mse:.6f}  mae: {mae:.6f}')

        with open('result_imputation.txt', 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'mse:{mse:.6f}, mae:{mae:.6f}\n\n')
