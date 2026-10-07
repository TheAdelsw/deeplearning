"""
GPT只保留了Transformer里的decoder块
同时去除交叉注意力 


"""

import torch
import torch.nn as nn





def attention(Q, K, V, mask = None):
    #Q K V 形状 [B, heads, seq_len, d_k]

    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-1, -2) / (d_k ** 0.5)    #[B, heads, seq_len, seq_len]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    scores = torch.softmax(scores, dim = -1)
    attn = scores @ V

    return attn





class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def split_heads(self, x):
        #x [B, seq, d_model]
        B, seq, _ = x.shape
        x = x.view(B, seq, self.heads, -1)  #[B, seq_len, heads, d_k]
        x = x.transpose(1, 2)               #[B, heads, seq_len, d_k]
        return x

    def forward(self, x, mask):
        #x [B, seq, d_model]
        B = x.size(0)

        Q = self.split_heads(self.W_q(x))    #[B, heads, seq, d_k]
        K = self.split_heads(self.W_k(x))    #[B, heads, seq, d_k]
        V = self.split_heads(self.W_v(x))    #[B, heads, seq, d_k]

        attn = attention(Q, K, V, mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(B, -1, self.heads * self.d_k)

        return self.W_o(attn)


"""
pre-LN 残差的旁路先LN 再层forward 再dropout

Block流程

attention

ffn

每层传入输入前先LN归一化 ffn中包含升维降维两个全连接



"""


class Block(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
        










if __name__ == '__main__':
    B, seq, d_model, heads = 2, 10, 256, 8
    blk = Block(heads=heads, d_model=d_model, d_ff=d_model * 4)
    blk.eval()                                #关掉dropout, 否则两次前向结果天然不同, 没法对比

    x = torch.randn(B, seq, d_model)
    mask = torch.tril(torch.ones(seq, seq))   #下三角1=准看, 上三角0=盖住

    out = blk(x, mask)
    print("输出形状:", out.shape)              #预期 [2, 10, 256]

    #causal验证: 篡改位置2之后的输入, 位置0、1的输出必须纹丝不动
    x2 = x.clone()
    x2[:, 2:] = 999.0                         #未来位置被污染
    out2 = blk(x2, mask)
    print("前2位置差异(应≈0):", (out[:, :2] - out2[:, :2]).abs().max().item())
    print("位置2差异(应>0):  ", (out[:, 2] - out2[:, 2]).abs().max().item())
