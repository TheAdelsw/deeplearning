#DDPM本体 训练对噪声的推理 预测的结果是从x0到xt的总噪声 然后由推导的数学公式进行反推t-1时刻的图
#数学骨架在scheduler里 网络架构在unet里

import torch
import torch.nn as nn

#指数滑动平均 将影子权重向每次最新的权重缓慢更新 使得训练平滑防止抖动
class EMA:
    def __init__(self, model, decay = 0.999):
        self.decay = decay
        self.shadow = {}    #影子权重 字典
        self.backup = {}    #本体备份 字典
        for name, p in model.named_parameters():    #该函数是返回计算图中所有的参数
            if p.requires_grad:     #筛选可以被更新梯度的参数
                self.shadow[name] = p.data.clone()


    @torch.no_grad()    
    def update(self, model):
        """
        每个step后调用一次
        """
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
                #mul_返回张量自己


    @torch.no_grad()
    def apply_shadow(self, model):

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])


    @torch.no_grad()
    def restore(self, model):
        """采样后: 本体回归, 继续训练"""
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])



#组装DDPM
class DDPM:
    def __init__(self, unet, scheduler, device = 'cuda'):
        self.unet = unet
        self.sch = scheduler
        self.device = device

    #训练 推理出噪声
    def p_losses(self, x0):
        """x0: [B,3,64,64] 干净图(归一化到[-1,1]) 返回标量loss"""
        B = x0.size(0)

        #每个批次的图各自抽取一个随机噪声等级 0~999 表示抽取的时间步
        t = torch.randint(0, self.sch.timesteps, (B,), device = self.device)

        #再抽一份噪声 一步求加噪图
        noise = torch.randn_like(x0)
        x_t = self.sch.q_sample(x0, t, noise)   #√我没看懂这里q_sample的数据流动 形状都是怎么样的 特别是sqrt_alphas_cumprod[t]这个代表了什么

        #由UNet网络来推理
        noise_pred = self.unet(x_t, t)

        #和真正的noise做MSE
        return nn.functional.mse_loss(noise_pred, noise)


    #采样 推理出前一步 即t -> t-1
    @torch.no_grad()
    def p_sample(self, x_t, t_idx):
        """
        从 x_t 走一步到 x_{t-1}
        公式: x_{t-1} = 1/√α_t·(x_t − β_t/√(1−ᾱ_t)·ε̂) + σ_t·z
        """
        t = torch.full((x_t.size(0),), t_idx, device=self.device, dtype=torch.long) #√这个函数是什么意思 其中传入的参数起到什么作用 t又是什么

        #网络推理出的噪声
        eps = self.unet(x_t, t)

        coef = self.sch.betas[t_idx] / self.sch.sqrt_1_minus_alphas_cumprod[t_idx]
        x = (x_t - coef * eps) / self.sch.alphas[t_idx].sqrt()

        if t_idx > 0:
            z = torch.randn_like(x_t)
            x = x + self.sch.betas[t_idx].sqrt() * z    
        return x

    #采样 完整从纯噪声图 推理到第0步图
    @torch.no_grad()
    def sample(self, n, img_size = 64, img_ch = 3):
        """从纯高斯噪声出发走完全部1000步, 返回n张图 范围在[-1,1]"""
        x = torch.randn(n, img_ch, img_size, img_size, device = self.device)
        for t_idx in reversed(range(self.sch.timesteps)):   
            x = self.p_sample(x, t_idx)

        return x


if __name__ == '__main__':
    from unet import UNet
    from scheduler import Scheduler

    device = 'cuda'
    sch = Scheduler(1000, device)
    unet = UNet(base_ch=64).to(device)
    ddpm = DDPM(unet, sch, device)

    # 测试1: 训练目标能算loss (未训练网络, loss应在1附近)
    x0 = torch.randn(4, 3, 64, 64, device=device)
    print("p_losses:", ddpm.p_losses(x0).item())

    # 测试2: 完整1000步采样跑通 (未训练, 出来的只会是噪声图, 属正常)
    # 大约要跑几十秒 —— 这就是将来生成真图的同一个循环
    x = ddpm.sample(2)
    print("sample输出:", x.shape, "均值:", x.mean().item())