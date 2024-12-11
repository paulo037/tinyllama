from ..config import Config
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, config: Config,  dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(
            config.n_embd, dtype=config.dtype))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
