"""



LayerNorm归一化 对每个词向量自己的 256 个维度做标准化 即对每个token的256维词向量空间做标准化
而BatchNorm 只是根据一个batch来 其会把不同的句子混在一起

推理数据流:

中文句子 [B, 32]假设每个批次共32个token
->词嵌入层Embedding 后 [B, 32, 256]

->经过三层权重独立的encoder层
每个encoder层内经过如下结构
    attention自注意力
    AddNorm归一化(AddNorm是残差相加再做依次LayerNorm)
    FFN加工
    AddNorm归一化

三层encoder后输出一个[B, 32, 256]张量 即富含语义的tokens

然后英文部分首先输入默认的 BOS特殊符号作为输入

进入三层权重独立的decoder层
每个decoder层内部经过如下结构
    因果掩码自注意力
    AddNorm
    与中文三层encoder输出的张量的交叉注意力
    AddNorm
    FFN加工

三层decoder后得到一个[B, 32, d_model]的输出 这个就是最终的答案 

将这个答案经过一个全连接输出头 转换为 logits[B, 32, V_size]
根据词表中最可能的词输出 然后将这个输出与一开始的BOS拼接为一段话 继续作为输入重复decoder的过程
直至输出EOS为止 



"""



import torch
import torch.nn as nn


from embedding import Embedding
from attention import MultiHeadAttention



#FFN前馈神经网络    共两层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x 形状为 [B, seq, d_model]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        out = self.dropout(x)
        return out
    




#残差层Res加归一化层LayerNorm
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_out):
        #x 子层的输入 主路  sublayer_out 子层的输出 旁路
        #先dropout子层输出 再残差相加 最后归一化
        sublayer_out = self.dropout(sublayer_out)
        out = self.norm(x + sublayer_out)
        return out



#encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)  

    def forward(self, x, pad_mask):
        #x [B, seq, d_model]  pad_mask挡住中文<pad>的权限地图
        #自注意力 Q K V 传入同一个x 
        ##attn返回(out, 权重)二元组, 这里只要out, 取[0]
        origin = x
        x = self.attn(x, x, x, pad_mask)[0]
        x = self.addnorm1(origin, x)

        origin = x
        x = self.ffn(x)
        out = self.addnorm2(origin, x)

        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)     #英文每个单词自查
        self.addnorm1 = AddNorm(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, heads)    #英文对照中文信息查
        self.addnorm2 = AddNorm(d_model, dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)
    

    def forward(self, x, memory, self_mask, cross_mask):
        #因果掩码自注意力
        x = self.addnorm1(x, self.self_attn(x, x, x, self_mask)[0] )
        
        #交叉注意力 Q=英文 K/V=memory, seq_q!=seq_k在这里发生 cross_mask挡中文pad
        # Q * K^T 再softmax得到 [B, heads, seq_q, seq_k]表示权重 再与V [B, heads, seq_k, d_model]
        #相乘得到[B, heads, seq_q, d_model]此时英文结合了中文的信息
        x = self.addnorm2(x, self.cross_attn(x, memory, memory, cross_mask)[0] )

        #FFN加工
        x = self.addnorm3(x, self.ffn(x))

        return x





#全部组装
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model = 256, heads = 8,
                 d_ff = 1024, num_layer = 3, max_len = 64, dropout = 0.1):
        super().__init__()

        #两套独立词嵌入 中文词表和英文词表
        self.src_emb = Embedding(src_vocab, d_model, max_len, dropout)  #中文源
        self.tgt_emb = Embedding(tgt_vocab, d_model, max_len, dropout)  #英文目标

        #必须用ModuleList而不是普通list 只有注册过 里面的参数才会被优化器看到 从而能计算梯度并更新
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])

        #输出层 给英文词表每个词打分 无softmax
        self.out = nn.Linear(d_model, tgt_vocab)


    def forward(self, src_ids, tgt_ids, src_masks, tgt_mask):
        #src_ids: [B, src_seq] 中文编号   tgt_ids: [B, tgt_seq] 英文编号

        B, T = tgt_ids.shape

        #对于英文自注意力 训练时需因果掩码
        #tril取下三角含对角 位置 j <= i才可以看
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_ids.device)).unsqueeze(0).unsqueeze(0)
        tgt_pad = tgt_mask.unsqueeze(1)     #[B,1,1,T] 广播到每个query行
        src_pad = src_mask.unsqueeze(1)     #[B,1,1,src_seq]


        #decoder自注意力 词不看未来
        self_mask = tgt_mask & causal
        cross_mask = src_pad

        #encoder三层
        x = self.src_emb(src_ids)
        for enc in self.encoders:
            x = enc(x, src_pad)

        memory = x      #此时形状为[B, src_seq, d_model]    富含了加工后的中文语义
        
        #decoder三层
        y = self.tgt_emb(tgt_ids)
        for dec in self.decoders:
            y = dec(y, memory, self_mask, cross_mask)

        
        #输出[B, T, tgt_vocab]  
        return self.out(y)
