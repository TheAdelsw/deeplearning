"""

记忆查询 


目前仅根据关键字匹配方式查找


"""

import core.memory.memory_library as memory_library



tool = {
    "type" : "function",
    "function" : {
        "name" : "recall_memory",
        "description" : "当你想要回忆一下有没有某件事或者你认为你可能知道某件事 可以调用这个工具",

        "parameters" : {
            "type" : "object",
            "properties" : {
                #type 是格式要求 description 用以给模型知晓各个参数的作用
                "keyword" : {"type" : "string", "description" : "想要回忆的内容的关键字"},
            },
            "required" : ["keyword"],     #两个都必须填
        }
    }
}


def recall_memory(keyword):
    result = memory_library.recall_memory(keyword)
    if result :
        return result
    else:
        return result


if __name__ == "__main__":
    out = recall_memory("生日")
    print(out)