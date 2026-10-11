"""
手动调用api

"""


import requests





from config import API_KEY, MODEL_NAME, URL, PERSONALITY_TRAIT








headers = {
    "Authorization" : "Bearer " + API_KEY
}




payload = {
    "model": MODEL_NAME,
    "messages": [
        #{"role": "system", "content": PERSONALITY_TRAIT},
        #{"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    "stream": False,

    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 100000,

    #"thinking": {"type": "disabled"},


}


MAX_ROUNDS = 3

#存储对话
def build_messages(history, role, text):
    history.append({"role" : role, "content" : text})
    if len(history) > MAX_ROUNDS * 2:
        del history[:len(history) - MAX_ROUNDS * 2]

    return [{"role" : "system", "content" : PERSONALITY_TRAIT}] + history



def call_llm(messages):

    send = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,

        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100000,

        "thinking": {"type": "disabled"},
    }

    while(True):
        #print("消息发送中...")
        response = requests.post(URL, headers=headers, json=send, timeout=60)

        if response.status_code != 200:
            # print(f"请求失败: {response.status_code}")
            # print(response.text)
            # return None
            continue

        data = response.json()
        msg = data["choices"][0]["message"]
        
        #history.append({"role" : "assistant", "content" : msg["content"]})


        return msg["content"]




"""

模型的响应体格式如下

{
    "choices": [
        {"message": {"role": "assistant", "content": "回复在这里"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 18, "completion_tokens": 42, "total_tokens": 60}
}



"""






if __name__ == '__main__':

    history = []


    while(True):
        user = input("<<< ")
        if not user:
            continue
        if user == "/exit":
            break

        messages = build_messages(history, "user" , user)
        out = call_llm(messages)
        if out is None:
            continue

        build_messages(history, "assistant", out)
        print(out)

    

    # response = requests.post(URL, headers=headers, json=payload, timeout=60)    #发送请求 并且返回接收到的所有信息

    # if response.status_code == 200: #状态码 200表示成功 401表示apikey错误 429表示频率超限 500表示服务器内部错误
    #     data = response.json()      #json转为py字典
    #     msg = data["choices"][0]["message"]
    #     print(msg["content"])
    # else:
    #     print(f"请求失败: {response.status_code}")
    #     print(response.text)



    