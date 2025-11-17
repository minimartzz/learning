"""
Implementing GPT-2 from scratch
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  
  def forward(self):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

@dataclass
class GPTConfig:
  block_size: int = 256
  vocab_size: int = 65
  n_layer: int = 6
  n_head: int = 6
  n_embd: int = 384


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # Define the overall architecture of the Transformer
    self.transformer = nn.ModuleDict({
      "wte": nn.Embedding(config.vocab_size, config.n_embd),            # Tokens -> Embeddings
      "wtp": nn.Embedding(config.block_size, config.n_embd),            # Positions -> Embeddings
      "h": nn.ModuleList([Block(config) for _ in range(n_layer)]),      # Hidden attention block
      "ln_f": nn.LayerNorm(config.n_embd)                               # Layer Normalisation
    })
    self.lm_head = nn.Linear(config.n_embd, config.n_head, bias=False)  # Prediction head