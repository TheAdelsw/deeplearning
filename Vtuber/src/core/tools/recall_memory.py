"""

记忆查询 


目前仅根据关键字匹配方式查找


"""

import core.memory.memory_library as memory_library



tool = {
    "type" : "function",
    "function" : {
        "name" : "recall_memory",
        "description" : "当你需要回忆对方的偏好 约定 或以前聊过的内容时调用此工具",

        "parameters" : {
            "type" : "object",
            "properties" : {
                #type 是格式要求 description 用以给模型知晓各个参数的作用
                "query" : {"type" : "string", "description" : "用自然语言描述想回忆的内容 不必是原文中的字词 例如 对方喜欢的颜色 对方的考试安排"},
            },
            "required" : ["query"],     #两个都必须填
        }
    }
}


def recall_memory(query):
    results = memory_library.recall_memory(query)
    if not results :
        #没有结果的时候返回对应未找到记忆的prompt
        return "未找到相关记忆"
    

    #记忆库只给数据 此处工具将数据整理并返回

    lines = [f"[{r['time']}] {r['content']}" for r in results]
    #分隔符.join(字符串列表) 把列表里所有字符串用分隔符串成一个新字符串
    return "\n" + "\n".join(lines)



if __name__ == "__main__":
    out = recall_memory("他今天吃了什么")
    print(out)