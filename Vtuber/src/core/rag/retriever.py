"""


检索器


输入一批候选文本和一个查询 输出按相似度排序的结果


不与外部产生耦合


"""

import numpy as np

from core.rag import embedder





def search(query, texts, top_k = 5, threshold = 0.35):


    """
    query 待查询的句子
    texts候选的文本



    返回 : [{'index': i, 'score': s}, ...] 按分数从高到低 可能为空列表
    
    """

    #如果本身没有记忆
    if not texts:
        return []

    

    #查询和候选拼一起一次性编码 省一次模型前向
    #必须同一个模型编码 
    vecs = embedder.encode([query] + texts)

    q_vec = vecs[0]      #第一条是查询向量 形状(512,)
    m = vecs[1:]         #剩下的是候选矩阵 形状(N, 512)


    #矩阵乘向量 
    scores = m @ q_vec   #形状(N,)

    #argsort返回"从小到大"的下标 [::-1]翻转成"从大到小"
    order = np.argsort(scores)[::-1]


    results = []
    for i in order:
        s = float(scores[i])


        if s < threshold:
            #循环是按分数降序 如果此时已经小于下限 则直接退出循环
            break
        results.append({"index": int(i), "score": s})

        if len(results) >= top_k:
            #如果检索的记忆已经到达上限 退出
            break



    #最终返回的格式 [{'index': 0, 'score': ...}, {'index': 1, 'score': ...}]
    
    return results





if __name__ == "__main__":
    #底层自测 直接跑:  cd 到 src 目录后  python -m core.rag.retriever
    texts = [
        "对方最喜欢的颜色是蓝色",
        "对方正在准备期末考试",
        "今天聊了星穹铁道",
    ]

    #杀招查询 子串匹配必死("色彩""偏好"都不在原文里) 语义检索要能命中第0条
    print(search("他对色彩有什么偏好", texts))

    #无关查询 期望空列表或全部被阈值拦下
    print(search("中午吃了什么", texts))

