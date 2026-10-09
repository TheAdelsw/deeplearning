"""
推理层


使用temperature调节logits之间的绝对差 TOPK 选取前k个候选词 增加输出的多样性

加入KV缓存节约算力
新token进来时 计算自己的Q K V
K = [缓存的旧K] + 新K
V = [缓存的旧V] + 新V

用新Q对整个K计算attention




"""
import os

import torch
import torch.nn as nn


from model import GPT, load_model
from bpe import BPE, EOT


VOCAB_SIZE = 4502
D_MODEL    = 128 
HEADS      = 4   
D_FF       = 512 
N_LAYERS   = 3   
BLOCK_SIZE = 128 

BPE_PATH = r'GPT\dataset\bpe.json'
MODEL_PATH = r'GPT\model\mygpt-1.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




def generate(model, bpe, prompt, max_tokens = 100, temperature = 0.8, top_k = 40):
    model.eval()
    generated = []


    #bpe转化为编号
    prompt_token = bpe.encode(prompt)

    #prompt_token 形状为[seq] 需要对齐变为[1, seq]    1即是批次的位置
    #转为张量
    ids = torch.tensor(prompt_token, dtype = torch.long, device = DEVICE).unsqueeze(0)


    #先将prompt全部给模型输出 再获取最后一个词的预测和当前的KV缓存
    logits, cache = model(ids)    #此时[B, seq, vocab_size]
    logits = logits[:, -1, :]     #作为循环生词的起点 此时形状为[B, vocabsize]
  


    for _ in range(max_tokens):
        #温度调节
        logits = logits / temperature
        
        #topk 只保留前k个候选
        v, _ =torch.topk(logits, top_k)     #前k名的分数 v形状[1, k]   
        logits[logits < v[ :, [-1]]] = float('-inf')     #第k名 = 前k里最小的 比它低的全部赋值为负无穷

        #按照概率选择
        probs = torch.softmax(logits, dim = -1)
        next_id = torch.multinomial(probs, num_samples = 1) #按照概率加权选择随机选择一个数 形状[1, 1]

        tid = next_id.item()    #单元素张量变为普通数字
        if tid == bpe.token2id[EOT]:
            break

        generated.append(tid)

        if ids.size(1) + len(generated) >= BLOCK_SIZE:
            break
        
        logits, cache = model(next_id, cache)   #logits形状为[B, 1, vocabsize]
        logits = logits[:, -1, :]

    return bpe.decode(generated)











if __name__ == '__main__':
    bpe = BPE(8192)
    bpe.load(BPE_PATH)

    model = GPT(VOCAB_SIZE, HEADS, D_MODEL, D_FF, N_LAYERS, BLOCK_SIZE).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"模型从 {MODEL_PATH} 加载成功")

    prompt = "这就是蓝银"

    out = generate(model, bpe, prompt, max_tokens=100, temperature=0.8, top_k=40)

    print(prompt + out)
