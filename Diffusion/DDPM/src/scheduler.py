#预计算 β/α/ᾱ 常量表，并提供 x_0到x_t 的闭式加噪

import torch


def linear_beta(timesteps, beta_start = 1e-4, beta_end = 0.02):
    #从 1e-4 均匀线性涨到 0.02 共 timesteps 个值 前期加噪很轻 后期越来越重
    return torch.linspace(beta_start, beta_end, timesteps) #返回一个一维张量



class Scheduler:
    def __init__(self, timesteps = 1000, device = 'cuda'):
        self.timesteps = timesteps


        betas = linear_beta(timesteps = timesteps)  #1000个加噪强度
        alphas = 1.0 - betas
        # 对应公式 ᾱ_t = α_1 × α_2 × ... × α_t  torch相应函数处理这个过程
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_1_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        #将数据放到显存上
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_1_minus_alphas_cumprod = self.sqrt_1_minus_alphas_cumprod.to(device)


    def q_sample(self, x0, t, noise):
        """闭式解加噪（对应公式 x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε） ε即为加到t步累计的噪声

        参数:
            x0:    [B,3,64,64] 干净图 已归一化到 [-1,1]
            t:     [B]         每张图各自随机抽的时间步，比如 [327, 891, 45, ...]
            noise: [B,3,64,64] 提前用 torch.randn 生成好的噪声
        返回:
            x_t:   [B,3,64,64] 加噪后的图
        """
        
        #高级索引 t [B] 将t的每个数作为一个索引返回出B个元素的张量 形状为[B]
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  #[B,1,1,1]
        s2 = self.sqrt_1_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return s1 * x0 + s2 * noise 
        #每个训练步中每个批次随机抽t 计算第t步的加噪图


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    device = 'cuda'
    scheduler = Scheduler(timesteps=1000, device=device)

    # 读一张训练集里的图，预处理方式和以后训练时保持一致
    img = cv2.imread(r"D:\source_data\anime-face\1.png")   #BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    img = cv2.resize(img, (64, 64))                        # [64,64,3] RGB
    img = torch.from_numpy(img).permute(2, 0, 1).float()   # [3,64,64]
    img = (img / 255.0) * 2 - 1                            # [-1,1]
    img = img.unsqueeze(0).to(device)                      # [1,3,64,64]
    
    # 挑几个有代表性的时刻：0 几乎无噪，999 纯噪声
    ts = [0, 100, 300, 600, 999]

    # 固定同一份噪声，这样几幅图之间唯一的区别就是加噪程度，方便对比
    noise = torch.randn_like(img)

    plt.figure(figsize=(15, 3))
    for i, t_val in enumerate(ts):
        t = torch.tensor([t_val], device=device)           # [1]
        x_t = scheduler.q_sample(img, t, noise)            # [1,3,64,64]

        # 反归一化 [-1,1] -> [0,1] -> [H,W,C] 交给 matplotlib
        show = x_t[0].permute(1, 2, 0).cpu().numpy()
        show = (show + 1) / 2
        show = show.clip(0, 1)

        plt.subplot(1, len(ts), i + 1)
        plt.imshow(show)
        plt.title(f"t={t_val}")
        plt.axis('off')
    plt.show()