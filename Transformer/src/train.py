"""

训练脚本

交叉熵摊平打分

logits 是 [B, max_len, Vocab]   答案是[B, max_len]

ignore_index=0 PAD 编号是 0 即忽略补齐部分经过网络产生的值

交叉熵损失要的形状是 [N, C] 打分行 + [N, ]答案  每个分数送进损失函数中-ln(x)中 查看选出正确答案的概率造成的损失

[N, C] 表示N个词每次预测C个可能词每个的logit [N,]答案是存储真正答案词的编号(假设idx) 然后直接在某个词的C个预测中取到 p[idx]
再送进损失函数中计算损失 

即答案的数字=取概率的下标



"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from data import TranslationDataset
from transformer import Transformer
from transformer import save_model
from transformer import load_model
from tokenizer import PAD

import os


#配置 超参数
BATCH = 64
EPOCHS = 60
LR = 3e-5
N_VAL = 1000        #留1000句做验证集, 看过拟合
CLIP = 1.0          #梯度裁剪阈值
CKPT = r"Transformer\model\checkpoint.pt"




def compute_loss(model, batch, criterion, device):
    #五件套搬上GPU

    src = batch["src_ids"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    tgt_out = batch["tgt_out"].to(device)
    src_mask = batch["src_mask"].to(device)
    tgt_mask = batch["tgt_mask"].to(device)

    logits = model(src, tgt_in, src_mask, tgt_mask)      #[B,32,Vocab]
    #此时的logits输出是每个词对下一个词的预测打分 推理时只需要考虑当前输入的最后一个词输出的概率即可


    #摊平 [B*32, Vocab]打分行   +   [B*32] 答案

    return criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("设备:", device)

    #数据处理
    ds = TranslationDataset(r"Transformer\data\cmn.txt")
    train_ds, val_ds = random_split(ds, [len(ds) - N_VAL, N_VAL])   #数据分为训练集和验证集
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)

    #模型 优化器 损失
    model = Transformer(src_vocab=len(ds.zh_tok), tgt_vocab=len(ds.en_tok)).to(device)
    print("参数量: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=ds.en_tok.token2id[PAD])
    #ignore_index 答案是PAD(编号0)的位置不计损失也不产生梯度


    #先加载模型 没有就重新训练
    if os.path.exists(CKPT):
        print("找到模型,加载中...")
        load_model(CKPT, model, optimizer)
    else:
        print("未找到模型, 重新开始训练")


    val_need = True
    if val_need :
        #验证
        model.eval()                           #dropout关 切换推理模式
        vtotal, vsteps = 0.0, 0
        with torch.no_grad():                  #不需要梯度, 省显存提速
            for batch in val_loader:
                vtotal += compute_loss(model, batch, criterion, device).item()
                vsteps += 1
        val_loss = vtotal / vsteps

        print(f"val_loss={val_loss:.3f} ppl={torch.exp(torch.tensor(val_loss)):.1f}")

        exit()








    #训练部分
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()       #使用dropout 训练模式用
            total, steps = 0.0, 0

            for step, batch in enumerate(train_loader, 1):  #enumerate(train_loader, 1)表示序号从1开始(不是索引下标)
                loss = compute_loss(model, batch, criterion, device)

                optimizer.zero_grad()              #清掉上一步的旧梯度
                loss.backward()                    #反向传播算新梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #防梯度爆炸
                optimizer.step()                   #更新权重

                total += loss.item()
                steps += 1
                if step % 100 == 0:
                    print(f"ep{epoch} step{step}/{len(train_loader)} loss={loss.item():.3f}")


            # #验证
            # model.eval()                           #dropout关 切换推理模式
            # vtotal, vsteps = 0.0, 0
            # with torch.no_grad():                  #不需要梯度, 省显存提速
            #     for batch in val_loader:
            #         vtotal += compute_loss(model, batch, criterion, device).item()
            #         vsteps += 1
            # val_loss = vtotal / vsteps

            # print(f"[epoch {epoch}] train_loss={total/steps:.3f} "
            #       f"val_loss={val_loss:.3f} ppl={torch.exp(torch.tensor(val_loss)):.1f}")

            #每30轮覆盖保存
            if epoch % 30 == 0:
                save_model(CKPT, model, optimizer)
    
    except KeyboardInterrupt:
        print("\n手动终止训练, 保存进度")
        save_model(CKPT, model, optimizer)
        


    

