from ..config import Config
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(
            config.n_embd, config.hidden_dim, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(
            config.n_embd, config.hidden_dim, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(
            config.hidden_dim, config.n_embd, bias=False, dtype=config.dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x)) * self.up_proj(x))
        return down_proj
