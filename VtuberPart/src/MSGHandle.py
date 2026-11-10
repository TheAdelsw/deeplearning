"""

消息处理的模块

由LLMBrain大脑初始化

处理各种消息格式 包括模型回复格式 工具调用格式




目前消息处理逻辑打算先简单写一个状态机

待输入  <->  模型思考  <->  tools工具调用

其中模型思考和tools工具调用的循环由此处处理 待输入状态即是循环终止 等待主循环用户输入
目前先写死一个上限 避免无限调用


"""

from config import PERSONALITY_TRAIT
from tools.tools import TOOLS, call_tool


class MSGHandle:
    def __init__(self, llmbrain):
        self.llmbrain = llmbrain
        self.call_tool = call_tool
        self.tools = TOOLS
        self.memory = [{"role" : "system", "content" : PERSONALITY_TRAIT}]

    #供用户主动输入消息的接口
    def UserSay(self, text, stream = False):
        msg = {"role" : "user", "content" : text}
        self.memory.append(msg)

        if stream == False:
            out = self.agent_loop()
            print(out)




        else:
            out = ""
            reply_stream = self.llmbrain.call_llm_stream(self.memory)
            for reply in reply_stream:

                content = reply['choices'][0]['delta'].get('content')


                if content:
                    print(content, end = "", flush = True)
                    out += content
            
            print()
            

        
        response = {"role" : "assistant", "content" : out}
        
        self.memory.append(response)

        return out




    def agent_loop(self, max_rounds = 10):
        #先发送给模型对应数据 如果需要调用工具就不断进行循环
        for _ in range(max_rounds) :
            all_reply = self.llmbrain.call_llm(self.memory, tools=self.tools)
            msg = all_reply['choices'][0]['message']

            #如果没有工具调用 直接返回
            if not msg.get('tool_calls'):
                return msg.get('content')


            #存储调用工具的请求记忆 需要回传
            self.memory.append({
                'role' : 'assistant',
                'content' : msg.get('content'), #模型在调用工具时也可能会有说一些话 不过一般是 None 空''
                'tool_calls' : msg['tool_calls']
                })

            #调用所有请求到的工具  注msg['tool_calls']返回的是一个列表 这个是列表for循环
            for tool in msg['tool_calls']:
                tool_name = tool['function']['name']
                tool_id = tool['id']
                tool_result = self.call_tool(tool_name=tool_name)
                self.memory.append({
                    'role' : 'tool', 
                    'tool_call_id' : tool_id, 
                    'content' : tool_result
                    })
        
        self.memory.append({
            'role' : 'system',
            'content' : '调用工具次数超限,已经自动停止,请收到此消息后下一轮回复停止调用工具'
        })
        
        all_reply = self.llmbrain.call_llm(self.memory, tools=self.tools)
        msg = all_reply['choices'][0]['message']
        return msg.get('content')
