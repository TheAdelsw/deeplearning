"""
BPE (Byte pair encoding) 分词器
从字符出发 将语料中高频出现的相邻token对合并为新的token 直到词表达到目标大小


先把文本切成块 BPE只在块内合并 


"""



import re
import json
from collections import Counter


UNK = "<UNK>"
EOT = "<|endoftext|>"   #文档分隔符, 也是生成时的停止信号



def pre_split(text):
    #预切分成块: 中文逐字 | 英文连续字母整块 | 其余(标点/符号)逐个
    #BPE只在块内合并, 块间永远不合并
    #\u4e00-\u9fff表示的是unicode里表意文字区 即中文字符区
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\s|[^\s]", text)






class BPE:
    def __init__(self, vocab_size = 8192):
        self.vocab_size = vocab_size

        self.merges = {}    #字符合并
        self.id2token = []
        self.token2id = {}
        
    def train(self, text, save_path = None):
        blocks = pre_split(text)    #["字符串1","字符串2"]
        seqs = [list(b) for b in blocks if len(b) > 1]    #每块拆分成字符列表 [["a", "b", "c"], [""...]]

        chars = sorted(set("".join(blocks)))     
        #set集合去掉重复元素 sorted将集合变为列表并排序 
        #语料是中英混合的 切分后重新拼接能够去除空格 并且获得中文每个字符 和英文26个字母
        
        self.id2token = [UNK, EOT] + chars
        self.token2id = {t : i for i, t in enumerate(self.id2token)}

        #要合并的次数 = 目标词表 - 已有的基础字符 - 特殊token
        for step in range(self.vocab_size - len(self.id2token)):
            #统计块内相邻对频次
            cnt = Counter()
            for s in seqs:      #如某个字符列表['t','h','e','h','e']
                for a, b in zip(s, s[1:]):
                    cnt[(a, b)] += 1
            if not cnt:                               #全语料没有可合并对了
                break                
            
            #然后将最高频字符对合并为新token 
            best, _ = cnt.most_common(1)[0]     #如返回('h','e'), 出现次数
            new_tok = best[0] + best[1]
            self.merges[best] = step            #merge["字符串"] = 合并规则序号step

            self.token2id[new_tok] = len(self.id2token)
            self.id2token.append(new_tok)

            #最后全语料重放
            for s in seqs:
                j = 0
                while j < len(s) - 1:
                    if (s[j], s[j + 1]) == best:
                        s[j:j + 2] = [new_tok]        #合并
                    j += 1
    
        print(f"词表大小: {len(self.id2token)} (基础字符{len(chars)} + 合并{len(self.merges)})")
        if save_path:
            self.save(save_path)


    def encode(self, text):
        ids = []
        for b in pre_split(text):
            s = list(b)
            while len(s) > 1:
                #收集块内所有"学过的合并对" 取rank最小的先合 即最先发现的合并规则
                cand = [(self.merges[(s[j], s[j + 1])], j)
                        for j in range(len(s) - 1)
                        if (s[j], s[j + 1]) in self.merges]
                if not cand:
                    break
                #此时cand列表内有若干个元组(组合规则的优先度, 对应当前字符列表的下标)
                _, j = min(cand)                      #rank最小 = 最该先合并
                s[j:j + 2] = [s[j] + s[j + 1]]

            for tok in s:      # 遍历块合并后剩下的每个 token
                ids.append(self.token2id.get(tok, self.token2id[UNK]))  # 有 id 用 id, 没有就 UNK 兜底
        return ids         

    def decode(self, ids):
        return "".join(self.id2token[i] for i in ids)

    
    def save(self, path):       #保留merges规则 用于编码
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.id2token, "merges": [list(k) for k in self.merges]}, f,
                      ensure_ascii=False)


    def load(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.id2token = d["vocab"]
        self.merges = {tuple(k): i for i, k in enumerate(d["merges"])}
        self.token2id = {t: i for i, t in enumerate(self.id2token)}        








if __name__ == "__main__":
    text = open(r"GPT\dataset\corpus2.txt", encoding="utf-8").read()
    bpe = BPE(vocab_size=8192)
    bpe.train(text, save_path=r"GPT\dataset\bpe.json")

    sample = "她看着他，不知道说什么才好。"
    ids = bpe.encode(sample)
    print("编码:", ids)
    print("还原:", bpe.decode(ids))          #应与原文一字不差

    #压缩率: BPE的战果 —— token数比字符数少多少
    all_ids = bpe.encode(text)
    print(f"压缩率: {len(all_ids)} token / {len(text)} 字符 = {len(all_ids)/len(text):.2f}")

    print("词表尾部(新长出来的token):", bpe.id2token[-15:])
