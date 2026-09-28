"""

自己建立词到编号的映射 

需要将一段话切成每个字或者每个单词 
同时需要有约定俗成的暗号 以便知道是否开始何时终止等信息

tokenizer的作用是将一段话变成一个个字符的编号序列

"""

import re
from collections import Counter


#四个特殊符号 

PAD = "<pad>"   #填充符 把长短不一的句子补齐到同一长度 才能拼成一个矩阵
BOS = "<bos>"   #begin of sequence: 句子开头的标记
EOS = "<eos>"   #end of sequence: 句子结束标记 翻译时模型输出它表示"翻完了"
UNK = "<unk>"   #unknown: 词表里没见过的词都用它兜底


#词处理器
class Tokenizer:
    def __init__(self, min_freq = 2):
        self.min_freq = min_freq    #出现次数低于它的词不进入词表 变为unk 防止词表过大而样本太少学不出有意义的向量
        self.token2id = {}          #词到编号 的字典
        self.id2token = []          #编号到词 的列表 列表的下标就是编号


    #把一句话切分成一串token
    def tokenize(self, text, lang):
        #中文无空格 英语有空格 将一个字符当作一个token
        if lang == "zh":    #中文
            #strip()去掉首尾空白, isspace()跳过句中所有空白字符
            return [ch for ch in text.strip() if not ch.isspace()]

        if lang == "en":    #英文
            #先转为消息 再用正则切分
            #r"[a-z']+"匹配连续的字母/撇号(能保住don't), |[.,!?;]匹配单个标点
            #例: "Don't stop!" -> ["don", "'", "t", "stop", "!"]
            return re.findall(r"[a-z']+|[.,!?;]", text.lower())

    #扫一遍语料 建立词表 然后永久存储
    def build(self, token_lists, lang):
        #Counter会自动统计每个token出现的次数 token_lists是很多句token列表的列表
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        #先放4个特殊符号, 它们编号固定为0,1,2,3 (0号留给PAD很重要, 后面padding和mask都要用)
        for tok in [PAD, BOS, EOS, UNK]:
            self.token2id[tok] = len(self.id2token)   #当前列表长度 = 下一个可用编号
            self.id2token.append(tok)

        #most_common()按出现次数从多到少返回, 排序后面频率只会更低, 低于阈值就可以直接停
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                break
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)

    def encode(self, tokens):
        #根据文字token返回对应编号
        #get(t, UNK的编号) 查不到的词(测试时遇到生词)返回UNK的编号
        return [self.token2id.get(t, self.token2id[UNK]) for t in tokens]

    def decode(self, ids):
        tokens = [self.id2token[i] for i in ids]
        #根据编号返回对应文字token 并且忽略特殊字符
        return "".join(t for t in tokens if t not in (PAD, BOS, EOS, UNK))

    def __len__(self):
        return len(self.id2token)




if __name__ == "__main__":
    tok = Tokenizer()
    zh = [tok.tokenize(s, "zh") for s in ["你好世界", "今天天气不错"]]
    en = [tok.tokenize(s, "en") for s in ["hello world", "don't stop!"]]
    tok.build(zh + en, "zh")

    # c = Counter()
    # c.update(["你好世界", "今天天气不错"])
    # print(c)
    # print("zh:", zh)
    # print("en:", en)

    print("词表大小:", len(tok))
    print("中文切分:", tok.tokenize("你好世界", "zh"))
    print("英文切分:", tok.tokenize("Don't stop!", "en"))
    print("编码:", tok.encode(tok.tokenize("你好世界", "zh")))
