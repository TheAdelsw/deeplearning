"""
模型组装


数据流                  输入数据形状         输出数据形状
BPE   token id           [B, seq]           
Embedding                [B, seq]        [B, seq, d_model]
Block * n                [B, seq, d_model]  [B, seq, d_model]
final LayerNorm          ...
权重使用词嵌入矩阵        [B, seq, d_model]  [B, seq, vocab_size]

logits即最终输出   [B, seq, vocab_size] 再temperature调节 softmax TOPK选出前k个最高概率   





"""


import torch
import torch.nn as nn

from embedding import Embedding
from block import Block





class GPT(nn.Module):
    def __init__(self, vocab_size, heads, d_model, d_ff, num_layers = 4, max_len = 128, dropout = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.heads = heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout

        self.embedding = Embedding(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            Block(heads, d_model, d_ff, dropout) for _ in range(num_layers)
        )

        self.final_layernorm = nn.LayerNorm(d_model)




    def forward(self, x):
        #x: [B, seq]     表示词的id

        B, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype = torch.bool, device = x.device)).unsqueeze(0).unsqueeze(0)

        x = self.embedding(x)

        for blk in self.blocks:
            x = blk(x, mask)
        
        x = self.final_layernorm(x)
        logits = x @ self.embedding.tok_emb.weight.T    #词嵌入矩阵的转置
        return logits




#save模型
def save_model(path, model, optimizer):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"模型已保存至{path}")


#load模型
def load_model(path, model, optimizer):
    ckpt = torch.load(path)     #torch.load读取pt文件
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"模型已从 {path} 成功载入")








if __name__ == '__main__':
    V = 2278                     #BPE的真实词表大小
    model = GPT(vocab_size=V, heads=8, d_model=256, d_ff=1024, num_layers=4, max_len=128)
    model.eval()

    x = torch.randint(0, V, (2, 16))        #随机id 模拟BPE编码后的输入
    logits = model(x)
    print("logits形状:", logits.shape)       #预期 [2, 16, 2278]

    #causal验证 篡改位置8之后, 前8个位置的logits必须纹丝不动
    x2 = x.clone()
    x2[:, 8:] = 5
    l2 = model(x2)
    

    print("前8位置差异(应≈0):", (logits[:, :8] - l2[:, :8]).abs().max().item())

    print("参数量:", sum(p.numel() for p in model.parameters()))



