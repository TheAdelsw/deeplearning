"""
各种工具统一放在一个文件中

tools工具基类 通过这个模块调用所有子工具 文件


"""




#此处import子工具
import tools.get_time as get_time



class ToolLibrary :
    def __init__(self):

        #一个存储多个工具描述字典的列表
        self.TOOLS = [
            get_time.tool,
        ]

        #存储工具调用方法
        self.TOOL_FUNCTIONS = {
            "get_time" : get_time.get_time
        }
    

    def call_tool(self, tool_name):
        if tool_name in self.TOOL_FUNCTIONS:
            return self.TOOL_FUNCTIONS[tool_name]()
        else :
            return "TOOL NOTFOUND"    




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

    
