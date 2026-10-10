"""
模拟对话过程

请忽略模型的垃圾输出

"""


import os

import torch
import torch.nn as nn


from model import GPT, load_model
from bpe import BPE
from generate import generate


VOCAB_SIZE = 4502
D_MODEL    = 128 
HEADS      = 4   
D_FF       = 512 
N_LAYERS   = 3   
BLOCK_SIZE = 128 

BPE_PATH = r'GPT\dataset\bpe.json'
MODEL_PATH = r'GPT\model\mygpt-1.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'








if __name__ == '__main__':
    bpe = BPE(8192)
    bpe.load(BPE_PATH)

    model = GPT(VOCAB_SIZE, HEADS, D_MODEL, D_FF, N_LAYERS, BLOCK_SIZE).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"模型从 {MODEL_PATH} 加载成功")

    

    try:
        while True:
            user = input("你说:").strip()
            if not user:
                continue
            if user == r"/exit":
                break
            out = generate(model, bpe, user, max_tokens=100, temperature=0.5, top_k=30)
            print("模型回答: " , user + out)

    except KeyboardInterrupt:
        print("结束对话")


