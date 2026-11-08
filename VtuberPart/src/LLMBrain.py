"""
整个流程的大脑 调用api接口

只接收和传递给MSGHandle文本消息


"""


from config import URL, API_KEY, MODEL_NAME

import requests
import json


headers = {
    "Authorization" : "Bearer " + API_KEY
}



class LLMBrain:
    def __init__(self):
        pass


    def call_llm(self, messages):

        send = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,

            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100000,

            "thinking": {"type": "enabled"},
        }

        while(True):
            response = requests.post(URL, headers = headers, json = send, timeout = 20)

            if response.status_code != 200:
                print("请求失败 status_code:", response.status_code)
                continue
            else:
                break

        data = response.json()
        msg = data['choices'][0]['message']

        return msg['content']



    #用了yield 转化为生成器函数 每次解析出一部分yield part出去 就暂停
    def call_llm_stream(self, messages):
        send = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,

            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100000,

            "thinking": {"type": "enabled"},
        }

        while(True):
            response = requests.post(URL, headers = headers, json = send, timeout = 20, stream = True)
            if response.status_code != 200:
                print("请求失败 status_code:", response.status_code)
                continue
            else :
                break

        for line in response.iter_lines():
            line = line.decode("utf-8")

            if not line:
                continue

            # print(f"此时chunk为: {chunk}")
            # print(f"此时line为: {line}")


            if "[DONE]" in line:
                break

            if line.startswith("data: "):
                line = line[6:]

            #以上三处if如果有一处漏写将会造成无法正确转化json格式从而报错


            #字符串转为字典
            chunk = json.loads(line)

            piece = chunk["choices"][0]["delta"].get("content")

            if piece:
                yield piece
            

        


            





