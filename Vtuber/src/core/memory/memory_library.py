"""


记忆库 接口暴露给工具调用


先使用文本记忆保存 后续改为sqlite


"""

#记忆的路径由config中读取
from core.config import MEMORY_PATH 


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



def recall_memory(keyword):
    """
    按照关键词查找
    最新的记忆最靠前
    """

    entries = _load_all()

    # reverse返回一个翻转后的列表
    result = [e for e in reversed(entries) if keyword in e['content']]

    return result



if __name__ == "__main__":

    pass




