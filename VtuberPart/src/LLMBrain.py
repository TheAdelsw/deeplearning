"""
整个流程的大脑 调用api接口

只接收和传递给MSGHandle文本消息


"""


from config import URL, API_KEY, MODEL_NAME

import requests



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
            data = response.json()
            msg = data['choices'][0]['message']

            return msg





