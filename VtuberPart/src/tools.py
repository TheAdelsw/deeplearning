"""
各种工具统一放在一个文件中



后续可能会分离优化


"""




from datetime import datetime


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



TOOL_FUNCTIONS = {
    "get_time" : get_time
}

TOOLS = [
    {
        "type" : "function",
        "function" : {
            "name": "get_time",
            "description": "获取当前的日期和时间。当对方询问时间、日期、今天几号、星期几、现在几点等一切与当前时间有关的问题时调用此工具",
            
            #参数部分
            "parameters" : {
                "type": "object",
                "properties": {},   #可以填的参数  格式如"city": {"type": "string", "description": "城市名，例如：北京"},
                "required": []      #必要的参数
            }

        }



    }
]



if __name__ == "__main__":

    print(TOOL_FUNCTIONS["get_time"]())


    
