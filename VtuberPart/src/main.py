"""
测试用 暂时的主程序



"""



from LLMBrain import LLMBrain
from MSGHandle import MSGHandle


from config import PERSONALITY_TRAIT




if __name__ == '__main__':
    brain = LLMBrain()
    msghandle = MSGHandle(brain)
    
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

        out = msghandle.UserSay(user)
        print(out)



