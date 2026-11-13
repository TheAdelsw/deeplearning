"""
测试用 暂时的主程序



"""



from core.LLMBrain import LLMBrain
from core.MSGHandle import MSGHandle
from core.tools.tools import ToolLibrary

from core.config import PERSONALITY_TRAIT



def main():
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



if __name__ == '__main__':

    main()



