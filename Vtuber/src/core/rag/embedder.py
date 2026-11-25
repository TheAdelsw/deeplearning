"""
RAG的词嵌入层 将token转为词向量


模型 bge-small-zh 约24M参数 CPU可跑

"""


#环境变量必须在导入模型库之前设置 否则库读不到 
import os

from core.config import HF_ENDPOINT, HF_HOME, HF_OFFLINE

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HOME"] = HF_HOME

#纯离线总闸 关闭一切对HF Hub的联网行为 包括更新检查
if HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"  #os.environ只装字符串 不装布尔值



import numpy as np
from sentence_transformers import SentenceTransformer


model = None

def load_model():

    #懒加载 仅在无模型对象时加载 后续返回同一个对象

    global model


    if model is None:
        #首次运行会从镜像下载权重文件到HF_HOME然后载入内存
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    return model


def encode(texts):
    #词嵌入编码器 
    #将输入的句子的每个token转为词向量 然后用一个全局的特殊token [CLS] 对全部token做自注意力 加权结合v
    #最终返回一个浓缩了所有句子信息的512维的词向量

    m = load_model()

    #normalize_embeddings=True 就是 L2归一化(模长变1) 归一化后点乘=余弦相似度

    #注 如果传入的是单字符串 那么返回的形状是(512) 如果传入的是字符串列表 返回形状是(n, 512)
    return m.encode(texts, normalize_embeddings=True)









if __name__ == "__main__":
    #手感实验 5句话覆盖三种关系 同义/同域反义/无关
    texts = [
        "我最喜欢的颜色是蓝色",
        "对方偏爱蓝色系",           #和第1句同义改写 期望相似度最高
        "今天中午吃了牛肉面",       #无关 期望垫底
        "他讨厌绿色的衣服",         #同域(颜色)但立场相反 期望中等
        "用户正在准备期末考试",     #无关 期望垫底
    ]

    vecs = encode(texts)
    print("向量矩阵形状:", vecs.shape)     #期望 (5, 512) 5句×512维

    #归一化后 点乘=余弦 矩阵乘自己的转置=两两相似度 5×5
    sim = vecs @ vecs.T
    print(np.round(sim, 2))




