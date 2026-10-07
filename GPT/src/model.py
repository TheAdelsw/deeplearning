"""
模型组装


数据流
token id [B, seq]
Embedding
Block * n
final LayerNorm
out
logits





"""


import torch
import torch.nn as nn

from bpe import BPE
from embedding import Embedding
from block import Block





class GPT(nn.Module):
    def __init__(self, vocab_size, heads, d_model, d_ff, num_layers = 4, max_len = 1000, dropout = 0.1):
        super().__init__()

        self.vocab = BPE(vocab_size)
        
        self.blocks = nn.ModuleList(
            Block(heads, d_model, d_ff, dropout) for _ in range(num_layers)
        )

        self.final_laynorm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)




    def forward(self, x):
        pass












