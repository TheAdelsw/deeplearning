"""
自己训练词嵌入层 根据网络上下载的语料数据


词嵌入 正弦位置编码 把token编号变成带位置信息的词向量

"""




import torch
import torch.nn as nn
import math




class Embedding(nn.Module):

    #         词表大小(收录的词)    向量维度  上下文长度  随机失活
    def __init__(self, vocab_size, d_model, max_len, dropout = 0.1):
        super().__init__()
        self.d_model = d_model

        #词嵌入矩阵 有vocab_size行 每行是该行编号下token的d_model维向量
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        #位置编码表 有max_len行 第pos行就是第pos个位置的位置向量
        pe = torch.zeros(max_len, d_model)
        
        #位置序列  0 1 2 3 ... 将形状 [max_len] 变为 [max_len, 1] 的列向量 便于广播  
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)


        #每个偶数维对应的频率 1/10000^(2i/d_model)
        #用指数-对数恒等式等价改写: 10000^(-2i/d) = exp(-ln(10000) * 2i/d)
        #arange(0, d_model, 2) -> [0, 2, 4, ...] 就是公式里的2i
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)      #[d_model/2]
            * (-math.log(10000.0) / d_model)
        )


        #广播相乘: [max_len,1] * [d_model/2] -> [max_len, d_model/2]
        #即每个位置 在每个频率上 的相位 pos/10000^(2i/d)
        pe[:, 0::2] = torch.sin(pos * div)   #偶数列填sin
        pe[:, 1::2] = torch.cos(pos * div)   #奇数列填cos


        #register_buffer: "跟模型走但不参与训练"的常量
        #会随state_dict保存、随.to(device)搬家, 但优化器永远不更新它
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        #x形状为 [B, seq_len]
        #查表得到[B, seq_len, d_model]
        x = self.tok_emb(x) * math.sqrt(self.d_model)   #乘以sqrt放大词向量
        x = x + self.pe[: x.size(1)]

        return self.dropout(x)

if __name__ == "__main__":
    emb = Embedding(vocab_size=100, d_model=64, max_len=50)
    x = torch.randint(0, 100, (2, 10))      #两句话 每句10个token编号

    out = emb(x)
    print("输出形状:", out.shape)             #预期 [2, 10, 64]

    #看一眼位置编码长什么样: 打印第0~4号位置的前8维
    print(emb.pe[:5, :8])
