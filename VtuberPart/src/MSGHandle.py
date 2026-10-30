"""

消息处理的模块

由LLMBrain大脑初始化



"""

from config import PERSONALITY_TRAIT



class MSGHandle:
    def __init__(self, llmbrain):
        self.llmbrain = llmbrain

        self.memory = [{"role" : "system", "content" : PERSONALITY_TRAIT}]

    #供用户主动输入消息的接口
    def UserSay(self, text, stream = False):
        msg = {"role" : "user", "content" : text}
        self.memory.append(msg)

        if stream == False:
            out = self.llmbrain.call_llm(self.memory)
            print(out)
        else:
            out = ""
            piece_stream = self.llmbrain.call_llm_stream(self.memory)
            for piece in piece_stream:
                print(piece, end = "", flush = True)
                out += piece
            
            print()
            
        
        response = {"role" : "assistant", "content" : out}
        
        self.memory.append(response)

        return out






