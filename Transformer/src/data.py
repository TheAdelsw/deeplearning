"""
数据处理部分

从https://www.manythings.org/anki/cmn-eng.zip 下载了中英语句
里面有些繁体污染数据 不过由于tokenizer的min_freq 其会被置为UNK

同时原句需要右移    句子太长直接扔了不截断
如 hi .

英文输入 BOS hi . PAD PAD ... PAD
英文答案 hi . EOS PAD PAD ... PAD


TranslationDataset中根据文件处理成中英词表


"""

import torch

from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer, PAD, BOS, EOS

MAX_LEN = 32        #一句中最多允许有32个token





class TranslationDataset(Dataset):
    def __init__(self, path):
        en_sents, zh_sents = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:              #脏行保护: 凑不齐两列的跳过
                    continue
                en_sents.append(parts[0])
                zh_sents.append(parts[1])


        
        #切词
        en_tokens = [Tokenizer().tokenize(s, "en") for s in en_sents]
        zh_tokens = [Tokenizer().tokenize(s, "zh") for s in zh_sents]

        #过滤 超过30个token的去除 给BOS 和 EOS 留位置
        keep = [i for i in range(len(en_sents))
                if len(en_tokens[i]) <= MAX_LEN - 2 and len(zh_tokens[i]) <= MAX_LEN - 2]
        en_tokens = [en_tokens[i] for i in keep]
        zh_tokens = [zh_tokens[i] for i in keep]

        #建两个词表
        self.en_tok = Tokenizer(min_freq=2)
        self.zh_tok = Tokenizer(min_freq=2)
        self.en_tok.build(en_tokens, "en")
        self.zh_tok.build(zh_tokens, "zh")


        #编号 并 右移
        pad_id = self.en_tok.token2id[PAD]          #PAD固定0号
        self.samples = []
        for en_t, zh_t in zip(en_tokens, zh_tokens):
            src = self.zh_tok.encode(zh_t)                       #中文原样
            src = [self.zh_tok.token2id[BOS]] + src + [self.zh_tok.token2id[EOS]]

            tgt = self.en_tok.encode(en_t)                       #英文
            tgt_in  = [self.en_tok.token2id[BOS]] + tgt + [self.en_tok.token2id[EOS]]
            tgt_out = tgt + [self.en_tok.token2id[EOS]]          #比输入右移一位


            #此时所有的词已经转换为编号形式


            #补齐到MAX_LEN
            def pad_to(ids, tok):
                return ids + [tok.token2id[PAD]] * (MAX_LEN - len(ids))

            self.samples.append((
                torch.tensor(pad_to(src, self.zh_tok)),                          #[32]
                torch.tensor(pad_to(tgt_in, self.en_tok)),                       #[32]
                torch.tensor(pad_to(tgt_out, self.en_tok)),                      #[32]
                torch.tensor(pad_to(src, self.zh_tok)).ne(pad_id).unsqueeze(0),  #[1,32]掩码
                torch.tensor(pad_to(tgt_in, self.en_tok)).ne(pad_id).unsqueeze(0)#[1,32]
            ))

        print(f"句对总数 {len(en_sents)} -> 保留 {len(self.samples)} | "
              f"中文词表 {len(self.zh_tok)} | 英文词表 {len(self.en_tok)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #返回一个字典
        return {"src_ids":  self.samples[idx][0],
                "tgt_in":   self.samples[idx][1],
                "tgt_out":  self.samples[idx][2],
                "src_mask": self.samples[idx][3],
                "tgt_mask": self.samples[idx][4]}



if __name__ == "__main__":
    ds = TranslationDataset(r"Transformer\data\cmn.txt")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    for k, v in batch.items():
        print(k, v.shape)                       #预期 src_ids[4,32] 掩码[4,1,32]

    i = 0
    print("中文:", ds.zh_tok.decode(batch["src_ids"][i].tolist()))
    print("英文输入:", ds.en_tok.decode(batch["tgt_in"][i].tolist(), sep=" "))
    print("英文答案:", ds.en_tok.decode(batch["tgt_out"][i].tolist(), sep=" "))