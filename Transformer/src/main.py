"""

Transformer 

中英翻译
数据流方向

字词token

词嵌入embedding层将其转为词向量 (同时加上位置编码)
中文经过三次独立encoder层(注意力 FFN 残差层和层归一化)  将中文词向量加工为语义丰富的语料
英文输入由<BOS>开始(训练时将英文答案右移 加上头尾作为输入 只加上尾作为输出)
经过三层decoder(自注意力 和中文词向量的交叉注意力 FFn 残差层和层归一化) 其中自注意力和交叉注意力的掩码由pad地图生成

最后输出每个词对下一个词的预测打分 由分最高的作为输出词

再拼接回输入 

循环往复直至输出EOS终止



"""


import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from data import TranslationDataset
from transformer import Transformer
from tokenizer import PAD, EOS, BOS

import os

CKPT = r"Transformer\model\checkpoint.pt"




def translate(zh_sentence):
    with torch.no_grad():
        #中文编码 进memory, 只算一次
        src_tokens = ds.zh_tok.tokenize(zh_sentence, "zh")  
        #切分为token

        src_ids = [ds.zh_tok.token2id[BOS]] + ds.zh_tok.encode(src_tokens) + [ds.zh_tok.token2id[EOS]]
        #token加上首尾特殊字符 再转为编号

        src = torch.tensor(src_ids).unsqueeze(0).to(device)     
        #转为张量再升维为[1, seq]   1作为batch

        src_mask = torch.ones(1, 1, src.size(1), dtype=torch.bool, device=device)  
        #单句无pad, 全True


        #decoder自回归推理
        out = torch.tensor([ [ds.en_tok.token2id[BOS]] ], device=device)   #[1,1]   编号形式

        for _ in range(40): #上限40个词 防止失控一直输出
            tgt_mask = torch.ones(1, 1, out.size(1), dtype=torch.bool, device=device)   #同样没有pad补齐
            logits = model(src, out, src_mask, tgt_mask)   #[1, t, V]
            next_id = logits[0, -1].argmax().item()
            if next_id == ds.en_tok.token2id[EOS]:  #说完了
                break
            
            out = torch.cat([out, torch.tensor([ [next_id] ], device=device)], dim=1)
            #[1, seq]   dim=1即在seq维度拼接
        
        return ds.en_tok.decode(out[0].tolist(), sep=" ")




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #重建数据集只是为了获取词表 词表的build过程是确定的
    #词表必须一致
    ds = TranslationDataset(r"Transformer\data\cmn.txt")

    model = Transformer(src_vocab=len(ds.zh_tok), tgt_vocab=len(ds.en_tok)).to(device)
    ckpt = torch.load(CKPT)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("输入中文开始翻译(直接回车退出)")
    while True:
        text = input("> ")
        if not text.strip():
            break
        print("=>", translate(text))
    