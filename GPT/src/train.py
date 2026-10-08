"""
训练脚本

数据流

数据loader
前向传播
CE交叉熵损失
反向传播计算梯度
裁剪梯度
更新


"""

import os 

import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader

from bpe import BPE
from data import FictionDataset
from model import GPT, save_model, load_model



#超参数

VOCAB_SIZE = 2278        #BPE真实词表大小
D_MODEL    = 256
HEADS      = 8
D_FF       = 1024
N_LAYERS   = 4
BLOCK_SIZE = 128        #即上下文长度; 位置编码的行数; 滑动窗口的大小
DROPOUT    = 0.1
BATCH_SIZE = 32
LR         = 3e-4        #Adam学习率 小GPT推荐
EPOCHS     = 40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'   
PATH = r'GPT\dataset\corpus.txt'
MODEL_PATH = r'GPT\model\mygpt-1.pt'

def main():
    bpe = BPE(8192)
    bpe.load(r"GPT\dataset\bpe.json")
    train_ds = FictionDataset(bpe, BLOCK_SIZE, PATH, split = 'train')
    val_ds = FictionDataset(bpe, BLOCK_SIZE, PATH, split = 'val')

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = True)

    model = GPT(VOCAB_SIZE, HEADS, D_MODEL, D_FF, N_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)
    print(f"设备: {DEVICE}  参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    ce = nn.CrossEntropyLoss()

    if os.path.exists(MODEL_PATH):
        print("找到模型权重 加载模型")
        load_model(MODEL_PATH, model, optimizer)
    else:
        print("未找到模型,训练新模型")


    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)

                #交叉熵损失需要摊平 N个样本×C类得分 [B, 128, V] 变为 [B*128, V], [B, 128]变为[B*128]
                loss = ce(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     #梯度裁剪 防止爆炸
                optimizer.step()

                total += loss.item() * x.size(0)

            train_loss = total / len(train_ds)


            if epoch % 20 == 0: #每一定轮数后保存和验证模型
                
                save_model(MODEL_PATH, model, optimizer)

                model.eval()
                vtotal = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        logits = model(x)
                        vtotal += ce(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item() * x.size(0)
                    
                    
                    val_loss   = vtotal / len(val_ds)
                    print(f"val {val_loss:.3f} | val ppl {math.exp(val_loss):.1f}")

            #困惑度ppl = exp(loss) 表示模型在多少个词中选择
            print(f"epoch {epoch:3d} | train {train_loss:.3f} ")


    except KeyboardInterrupt:
        print("\n手动终止训练, 保存进度")
        save_model(MODEL_PATH, model, optimizer)




if __name__ == '__main__':
    main()



