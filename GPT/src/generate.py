"""
推理层


使用temperature调节logits之间的绝对差 TOPK 选取前k个候选词 增加输出的多样性

加入KV缓存节约算力
新token进来时 计算自己的Q K V
K = [缓存的旧K] + 新K
V = [缓存的旧V] + 新V

用新Q对整个K计算attention




"""
import torch
import torch.nn as nn


from model import GPT, load_model



VOCAB_SIZE = 4502
D_MODEL    = 128 
HEADS      = 4   
D_FF       = 512 
N_LAYERS   = 3   
BLOCK_SIZE = 128 
DROPOUT    = 0.1 
BATCH_SIZE = 32  

MODEL_PATH = r'GPT\model\mygpt-1.pt'















if __name__ == '__main__':
    pass


