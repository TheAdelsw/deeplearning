"""
注意力模块


多头 交叉 注意力

"""

import torch

import torch.nn as nn



def attention(Q, K, V, mask = None):
    #Q K V形状均为 [B, heads, seq_len, d_k]     后两个是一句话的token长度以及每个头的d_k维
    #mask:1表示可以看该词 0表示禁止看该词     通过广播到分数矩阵来实现

    d_k = Q.size(-1)    #读取参数最后一维

    #点乘打分 获取应该要结合V的权重
    #[.., seq_q, d_k] @ [.., d_k, seq_k] -> [.., seq_q, seq_k]
    #即[B,Heads, token数, token数]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        #根据mask的位置 mask里 值==0的位置 对应同位置scores填-inf

    attn = torch.softmax(scores, dim = -1)  
    #对最后一维即[.., seq_q, seq_k] 的seq_k做softmax 对应于q中每个词向其他词的关系权重 

    return torch.matmul(attn, V), attn
    #V形状[..., seq_k, d_k] 返回[.., seq_q, d_k]




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):

        super().__init__()
        assert d_model % heads == 0     #必须整除 每个头分到d_model/heads维

        self.heads = heads
        self.d_k = d_model // heads

        #Q K V各自拥有一套全连接输出权重 所有词共用同一套权重
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)      #多头拼接后融合加工的输出层


    def split_heads(self, x):
        #x: [B, seq, d_model] -> [B, heads, seq, d_k]
        B, seq, _ = x.shape
        #view: 把d_model维拆成heads*d_k两段; transpose: 把heads维提到seq前面
        return x.view(B, seq, self.heads, self.d_k).transpose(1, 2)


    def forward(self, Q_in, K_in, V_in, mask = None):
        #输入的x形状为 [B, seq, d_model]
        B = Q_in.size(0)

        #先经过权重矩阵后再拆头
        Q = self.split_heads(self.W_q(Q_in))
        K = self.split_heads(self.W_k(K_in))
        V = self.split_heads(self.W_v(V_in))

        #每个头算注意力
        out, attn = attention(Q, K, V, mask)

        #拼回多头: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        #transpose后内存不连续, view要求连续内存, 所以先contiguous()
        out = out.transpose(1, 2).contiguous().view(B, -1, self.heads * self.d_k)

        #获取注意力后的总注意力经过全连接再加工信息
        return self.W_o(out), attn





if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=64, heads=8)
    x = torch.randn(2, 10, 64)               #两句话 各10个token 每个token 64维

    out, attn = mha(x, x, x)                 #自注意力: 三个传同一个x
    print("输出形状:", out.shape)              #预期 [2, 10, 64]
    print("注意力形状:", attn.shape)           #预期 [2, 8, 10, 10]
    print("每行加和:", attn[0, 0].sum(dim=-1))  #预期全是1.0


