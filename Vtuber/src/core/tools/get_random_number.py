"""

随机数工具

"""

import random

#schema 模型可调用工具的注册表 properties 的键名 = 参数名 模型照着填
tool = {
    "type" : "function",
    "function" : {
        "name" : "get_random_number",
        "description" : "生成一个指定范围内的随机整数 当对方想要随机数 抽签 掷骰子 帮忙做随机决定时调用此工具",

        "parameters" : {
            "type" : "object",
            "properties" : {
                #type 是格式要求 description 用以给模型知晓各个参数的作用
                "min" : {"type" : "integer", "description" : "闭范围下限 例如:1"},
                "max" : {"type" : "integer", "description" : "闭范围上限 例如:100"},
            },
            "required" : ["min", "max"],     #两个都必须填
        }
    }
}


def get_random_number(min, max):
    #错误当结果喂回：模型能读懂这串字，下一轮自己纠正参数重调
    if min > max:
        return "参数错误:min 不能大于 max"

    return random.randint(min, max)
