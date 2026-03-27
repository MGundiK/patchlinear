"""
Reversible Instance Normalisation (RevIN).
Source: xPatch (unchanged).

Handles distribution shift between training and test windows by normalising
each input instance to zero mean / unit variance, then reversing the
transformation after the model prediction.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = True, subtract_last: bool = False):
        """
        Parameters
        ----------
        num_features  : number of input channels C
        eps           : numerical stability constant
        affine        : if True, learn per-channel scale and shift after norm
        subtract_last : if True, subtract the last timestep instead of the mean
        """
        super().__init__()
        self.num_features  = num_features
        self.eps           = eps
        self.affine        = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [B, L, C]
        mode : 'norm' | 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        raise NotImplementedError(f"mode must be 'norm' or 'denorm', got '{mode}'")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.stdev = (
            x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).sqrt().detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x
