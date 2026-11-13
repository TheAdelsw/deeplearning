"""

记忆保存工具

由模型自己决定要不要保存记忆



为了解耦 工具调用只负责传参和调用工具 
如有使用其他包的API 不考虑其他问题如路径格式啥的

"""



#引入记忆库的方法 
import core.memory.memory_library as memory_library



#描述
tool = {
    "type" : "function",
    "function" : {
        "name" : "save_memory",
        "description" : "当你认为某些内容重要 想要记忆某些内容的时候 调用此工具",

        "parameters" : {
            "type" : "object",
            "properties" : {
                #type 是格式要求 description 用以给模型知晓各个参数的作用
                "content" : {"type" : "string", "description" : "想要记忆的主要内容"},
            },
            "required" : ["content"],     #两个都必须填
        }
    }
}



def save_memory(content):
    memory_library.save_memory(content=content)
    return "记忆保存成功"
