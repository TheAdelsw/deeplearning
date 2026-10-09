"""
滑动窗口训练 给定一段超长文本 在此不断截取一个定长的窗口 通过下移错位的方法训练

Dataset创建时传入构建好的BPE编码器

"""


import torch

from torch.utils.data import DataLoader, Dataset
from bpe import BPE


class FictionDataset(Dataset):
    def __init__(self, bpe, window_size, path, stride = None, split = 'train', val_ratio = 0.1 ):
        super().__init__()
        #stride 右移步长
        #split 将数据分割成train val用以创建验证集

        self.window_size = window_size
        self.path = path

        self.bpe = bpe


        #读取小说文本
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()     #一次性读入内存 

        #将text编码
        ids = bpe.encode(text)      #id 列表 
        ids = torch.tensor(ids, dtype = torch.long)     #转张量 long类型


        #切分训练集验证集
        n = int((len(ids)) * (1 - val_ratio))
        ids = ids[:n] if split == 'train' else ids[n:]

        if stride is None:
            stride = window_size // 2
        
        xs, ys = [], []
        
        #窗口划分 第i个窗口 x = ids[i:i+window_size], y = ids[i+1:i+window_size+1]
        for i in range(0, len(ids) - window_size - 1, stride):
            xs.append(ids[i : i + window_size])
            ys.append(ids[i + 1 : i + 1 + window_size])

        #现在xs是一堆[window_size]大小的小张量的集合 通过stack叠在一起
        self.x = torch.stack(xs)    #[样本数, window_size]
        self.y = torch.stack(ys)    #形状如上 内容右移一位

        print(f"[{split}] 样本数: {len(self.x)}  (语料{len(ids)} token, 窗口{window_size}, 步长{stride})")


    def __len__(self):  #获取样本数
        return self.x.size(0)



    def __getitem__(self, idx): #应返回单个样本的数据
        return self.x[idx], self.y[idx]







if __name__ == '__main__':
    bpe = BPE(8192)
    bpe.load(r'GPT\dataset\bpe.json')     #加载上次训练好的词表 —— 用load, 千万别再train(会覆盖bpe.json)

    train_ds = FictionDataset(bpe, window_size = 128, path = r'GPT\dataset\corpus2.txt', split = 'train')
    val_ds   = FictionDataset(bpe, window_size = 128, path = r'GPT\dataset\corpus2.txt', split = 'val')

    x, y = train_ds[0]                    #抽一个样本看
    print("x形状:", x.shape, " y形状:", y.shape)        #都是 [128]
    print("x前10:", x[:10].tolist())                   #tolist: 张量转普通列表好打印
    print("y前10:", y[:10].tolist())                   #观察: y就是x左移一位(去掉第一个, 多出下一个)
    print("x还原:", bpe.decode(x[:20].tolist()))        #解码前20个token回文字, 人眼验收

    loader = DataLoader(train_ds, batch_size = 16, shuffle = True)   #shuffle: 每个epoch打乱抽样顺序
    xb, yb = next(iter(loader))           #next(iter(...)): 手动抽一个batch看
    print("一个batch:", xb.shape, yb.shape)             #预期 [32, 128]
    print("batch内第一个样本的前10个词:", bpe.decode(xb[0, :10].tolist()))




