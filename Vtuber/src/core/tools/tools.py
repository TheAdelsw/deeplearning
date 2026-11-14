"""
各种工具统一放在一个文件中

tools工具基类 通过这个模块调用所有子工具 文件


"""

import json



#此处import子工具
import core.tools.get_time as get_time
import core.tools.get_random_number as get_random_number
import core.tools.save_memory as save_memory
import core.tools.recall_memory as recall_memory


class ToolLibrary :
    def __init__(self):

        #一个存储多个工具描述字典的列表 用以传给模型 使模型得知可调用的工具
        self.TOOLS = [
            get_time.tool,
            get_random_number.tool,


            save_memory.tool,
            recall_memory.tool
        ]

        #存储工具调用方法
        self.TOOL_FUNCTIONS = {
            "get_time" : get_time.get_time,
            "get_random_number" : get_random_number.get_random_number,

            "save_memory" : save_memory.save_memory,
            "recall_memory" : recall_memory.recall_memory,
        }
    

    def call_tool(self, tool_name, tool_args_json = None):
        func = self.TOOL_FUNCTIONS.get(tool_name)

        
        if func is None:
            print(f"[工具调用] {tool_name} 失败 不存在此工具"  )
            
            return f"TOOL NOTFOUND: {tool_name}"




        try :
            #获取参数   json格式转为字典
            args = json.loads(tool_args_json) or {}
            #   **args解包运算传参 str将所有结果变为字符串以便传输回模型
            result = str(func(**args))
            print(f"[工具调用] {tool_name} , 传入参数{args} result: {result}")
            return result

        except json.JSONDecodeError as e:

            print(f"[工具调用] {tool_name} 失败 json参数格式错误"  )

            #如果模型给的json格式出错 原文返回告诉模型
            return f"参数解析出错: {e}, 原始参数为: {tool_args_json}"

        except Exception as e:
            print(f"[工具调用] {tool_name} 失败 未知错误"  )
            #其他问题出错也返回错误结果给模型
            return f"TOOL ERROR: {e}"


# TOOL_FUNCTIONS = {
#     "get_time" : get_time.get_time
# }




# #一个存储多个工具描述字典的列表
# TOOLS = [
#     get_time.tool,
# ]


# def call_tool(tool_name):
#     if tool_name in TOOL_FUNCTIONS:
#         return TOOL_FUNCTIONS[tool_name]()
#     else :
#         return "TOOL NOTFOUND"



#模型的tool_calls格式如下
#[{'index': 0, 'id': 'call_00_ET_SWmgMHfLdaxxQqbuMO2X7803', 'type': 'function', 'function': {'name': 'get_time', 'arguments': '{}'} },	]

#需要给模型的回复如下
#{'role': 'tool', 'tool_call_id': 'call_0_abc123', 'content': '2026-11-10 15:30:00'}



if __name__ == "__main__":

    # print(TOOL_FUNCTIONS["get_time"]())
    toollibrary = ToolLibrary()

    print(toollibrary.call_tool(tool_name="get_time"))

    
