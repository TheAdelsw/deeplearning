"""


记忆库 接口暴露给工具调用


先使用文本记忆保存 后续改为sqlite


"""

#记忆的路径由config中读取
from core.config import MEMORY_PATH 



#引用RAG的检索器
from core.rag import retriever

import json 
import os

from datetime import datetime




def _load_all():
    #读取记忆文件 返回条目 文件不存在或损坏时返回空列表

    if not os.path.exists(MEMORY_PATH):
        return []
    
    #utf-8编码打开
    with open(MEMORY_PATH, 'r', encoding = 'utf-8') as f:
        try :
            return json.load(f)
        except json.JSONDecodeError:
            print("[记忆库] 文件损坏")
            return []


def _save_all(entries):
    #将整个对话条目写入文件 
    # ensure_ascii=False：中文原样存储 防止被转为乱码

    with open(MEMORY_PATH, 'w', encoding = 'utf-8') as f:
        json.dump(entries, f, ensure_ascii = False, indent = 2)








def save_memory(content):
    """
    存储一条记忆 自动附上时间
    
    """
    
    entries = _load_all()   #获取了指向记忆库的标签名即指针
    entries.append({
        'time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'content' : content,
    })
    _save_all(entries)
    return True



def recall_memory(query):
    """
    语义检索记忆
    RAG
    返回最相关的几条记忆
    可能为空列表


    """

    entries = _load_all()

    if not entries :
        return []


    #获取记忆中的文本
    texts = [e['content'] for e in entries]

    
    hits = retriever.search(query, texts)

    result = [entries[i['index']] for i in hits]

    return result




def inject_memory(query):
    #自动注入RAG检索 将所有相关信息整合为一条system信息
    #查不到时返回空列表


    results = recall_memory(query)

    #RAG索引无内容时返回空列表
    if not results:
        return []

    lines = [f"[{r['time']}] {r['content']}" for r in results]

    #将所有内容整合
    content = (
        "以下是记忆库中与当前话题可能相关的记忆 回复时可以参考:\n"
        + "\n".join(lines)
    )

    out_dic = [{"role" : "system", "content" : content}]

    return out_dic



if __name__ == "__main__":

    pass




