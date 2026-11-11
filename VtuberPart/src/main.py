"""
测试用 暂时的主程序



"""



from LLMBrain import LLMBrain
from MSGHandle import MSGHandle
from tools.tools import ToolLibrary

from config import PERSONALITY_TRAIT




if __name__ == '__main__':
    brain = LLMBrain()
    toolkit = ToolLibrary()
    msghandle = MSGHandle(brain, toolkit)
    
    # print(msghandle.memory)
    while(True):
        user = input("<<< ")
        if not user :
            continue
        if user == "/exit":
            break
        if user == "/memory":
            print(msghandle.memory)
            continue

        out = msghandle.UserSay(user, stream=True)
        



