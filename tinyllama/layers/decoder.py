from ..config import Config
import torch.nn as nn
from .rms_norm import RMSNorm
from .feed_forward import FeedForward
from .group_query_attention import GroupQueryAttention



class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.attn = GroupQueryAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)

    def forward(self, x, pos_emb, mask=None):
        n_1 = self.norm1(x)
        h = self.attn(n_1, pos_emb)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x
