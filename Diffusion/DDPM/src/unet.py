import math
import torch
import torch.nn as nn


#时间嵌入模块 让模型知道自己现在的时间步 从而决定噪点的修改强度
#使用与Transformer同样的位置编码 单独时间索引信息太轻 使用MLP增强信息
"""

PE(t, 2i)   = sin( t / 10000^(2i/d) )
PE(t, 2i+1) = cos( t / 10000^(2i/d) )

"""
class TimeEmbedding(nn.Module):
    def __init__(self, base_ch = 64, sinu_dim = 128):
        super().__init__()
        self.sinu_dim = sinu_dim
        emb_dim = base_ch * 4   #最终的时间向量维度为256

        self.mlp = nn.Sequential(
            nn.Linear(sinu_dim, emb_dim),
            nn.SiLU(),  #DDPM的激活函数
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, t):
        #t 是时间步张量 [B] 每个批次都有各自的时间步
        half = self.sinu_dim // 2   # 64，一半给 sin 一半给 cos
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )                           # [64]

        # 每个频率和每个 t 相乘得到相位：广播成 [B 64] 的矩阵
        args = t.float()[:, None] * freqs[None, :]   # [B 64]

        #没有看懂freqs和args到底是什么 你给我详细讲一下这个时间编码

        # sin 和 cos 各 64 维拼起来 -> [B 128]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)           # [B 256] 注意此时未使用激活函数

#残差块 输入in通道 输出out通道 时间向量嵌入维度同out通道数 共2次卷积1次时间维度全连接
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim = 256):
        super().__init__()
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, in_ch) #group归一化 将每个批次的通道分为8个组 平均数值
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)   # 3x3卷积 step=1 padding=1保持尺寸
        # 时间注入通道：256维时间向量 -> out_ch 维 之后广播加到特征图
        self.t_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # 通道数变了就加 1x1 卷积对齐维度；没变就用 Identity 原样通过
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()   #identity是原样返回输入
    
    def forward(self, x, t_emb):
        # x: [B,in_ch,H,W]  t_emb: [B,256]
        h = self.conv1(self.act(self.norm1(x)))               # [B,out_ch,H,W]
        
        t = self.t_proj(self.act(t_emb))[:, :, None, None]  #此处将二维的时间编码[B,ch]变为[B,ch,1,1]

        h = h + t   #时间编码注入

        h = self.conv2(self.act(self.norm2(h))) 
        """
        此处先将输入x初加工为h隐藏层数据 再注入时间信息 这是为了统一在out维度下对齐 方便注入后进行再加工
        """

        return h + self.skip(x) #残差 但是原x需要经过卷积对齐通道 但是仍然能高效传播梯度


#自注意力 输入ch通道 经过自注意力架构输出同样的ch通道的数据 但是每个通道的每个位置都结合了其他所有位置的信息
class SelfAttention(nn.Module):
    """自注意力：每个位置直接查询全图所有位置，实现全局信息交换
    只用在低分辨率层(16x16/8x8)，高分辨率层算不动也没必要
    
    """

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)

        # 1x1卷积一次性算出 Q K V 三份特征 C -> 3C 再分开
        self.qkv = nn.Conv2d(ch, ch * 3, 1) #我有疑问 算q k v的时候不是都是将原始数据通过一个MLP层来算的吗
        
        # 输出投影：把 从别处收集来的信息 融合回本通道
        self.proj = nn.Conv2d(ch, ch, 1)        

        # 关键技巧：缩放因子初始化为 0
        # 训练初期注意力等于不存在(输出0)，网络先当纯卷积网学，再逐渐学会用注意力
        # 零初始化是 diffusion/StyleGAN 系常用的稳定训练手段
        self.gamma = nn.Parameter(torch.zeros(1))   #这边的nn.Parameter函数 将其放入参数列表 参与梯度更新
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        h = self.norm(x)    #此处经过norm修改原始数据 被设计成这样 主流做法

        q, k, v = self.qkv(h).chunk(3, dim=1)   # 各 [B, C, H, W]
        #chunk将通道均分吗

        q = q.view(B, C, N).permute(0, 2, 1)    # [B, N, C] 每个位置一行=它的"提问"
        k = k.view(B, C, N)                     # [B, C, N] 每个位置一列=它的"简历"
        v = v.view(B, C, N).permute(0, 2, 1)    # [B, N, C] 每个位置一行=它可提供的"信息

        #求得相应的注意力权重(此时还没有注入信息v) 另外(C ** -0.5)也是transformer里论文防止参数爆炸的做法
        attn = q @ k * (C ** -0.5)
        #化为概率分布 即每个位置的信息分配比例 决定了要融合某位置的多少信息
        attn = attn.softmax(dim=-1)     #此时attn 是B批次的 N * N 矩阵

        out = attn @ v
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  #先变为B C N 再将最后的N 变为H W

        return x + self.gamma * self.proj(out)
        # 这边的proj 通过1*1卷积类似一层全连接线性映射 即Transformer中对注意力得来的消息精加工



#组装为完整的UNet
class UNet(nn.Module):
    def __init__(self, base_ch = 64, img_ch = 3, emb_dim = 256):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8  # 64/128/256/512
        self.time_emb = TimeEmbedding(base_ch)
        self.act = nn.SiLU()

        #将3通道先化为64通道
        self.stem = nn.Conv2d(img_ch, c1, 3, padding = 1)

        #下采样 通道翻倍 分辨率减半  每层2个残差块 适量的增大模型 同时时间向量反复注入作为全局变量
        self.down1_1 = ResBlock(c1, c1, emb_dim)    #64     分辨率 64
        self.down1_2 = ResBlock(c1, c1, emb_dim)
        self.down1 = nn.Conv2d(c1, c1, 4, stride = 2, padding = 1)    #  分辨率32

        self.down2_1 = ResBlock(c1, c2, emb_dim)   # 通道提升放这级第一个块 64->128
        self.down2_2 = ResBlock(c2, c2, emb_dim)
        self.down2 = nn.Conv2d(c2, c2, 4, stride = 2, padding = 1)    # 分辨率16

        self.down3_1 = ResBlock(c2, c3, emb_dim)    # 128->256
        self.down3_2 = ResBlock(c3, c3, emb_dim)
        #此时开始有全局视野
        self.attn3 = SelfAttention(c3)   #通道256 分辨率16
        self.down3 = nn.Conv2d(c3, c3, 4, stride = 2, padding = 1)    # 分辨率 8

        self.down4_1 = ResBlock(c3, c4, emb_dim)
        self.down4_2 = ResBlock(c4, c4, emb_dim)
        self.attn4 = SelfAttention(c4)             # 8x8

        #最后U形底层 信息最浓缩 再来二层残差块
        self.mid1 = ResBlock(c4, c4, emb_dim)
        self.mid_attn = SelfAttention(c4)
        self.mid2 = ResBlock(c4, c4, emb_dim)



        #上采样
        # Upsample 最近邻插值放大 + 3x3卷积(通道 c->c/2)消除插值锯齿

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),    #分辨率8->16
            nn.Conv2d(c4, c3, 3, padding = 1)   #通道512->256
        )
        self.up3_1 = ResBlock(c3 * 2, c3, emb_dim)    #为了融合同级的上采样和下采样信息
        self.up3_2 = ResBlock(c3, c3, emb_dim)
        self.up_attn3 = SelfAttention(c3)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.Conv2d(c3, c2, 3, padding = 1))       # 分辨率16->32
        self.up2_1 = ResBlock(c2 * 2, c2, emb_dim)
        self.up2_2 = ResBlock(c2, c2, emb_dim)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.Conv2d(c2, c1, 3, padding = 1))       # 32->64
        self.up1_1 = ResBlock(c1 * 2, c1, emb_dim)
        self.up1_2 = ResBlock(c1, c1, emb_dim)

        self.head_norm = nn.GroupNorm(8, c1)
        self.head = nn.Conv2d(c1, img_ch, 3, padding = 1)


    def forward(self, x, t):
        # x [B,3,64,64] 加噪图   t [B] 时间步   输出 [B,3,64,64] 噪声预测
        t_emb = self.time_emb(t)

        #下采样
        h = self.stem(x)    #通道3->64
        s1 = self.down1_2(self.down1_1(h, t_emb), t_emb)
        h = self.down1(s1)  #

        s2 = self.down2_2(self.down2_1(h, t_emb), t_emb)
        h = self.down2(s2)                                 # [128 16]

        s3 = self.down3_2(self.down3_1(h, t_emb), t_emb)
        s3 = self.attn3(s3)
        h = self.down3(s3)                                 # [256 8]

        h = self.down4_2(self.down4_1(h, t_emb), t_emb)
        h = self.attn4(h)                                  # [512 8]

        # 瓶底
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        h = self.up4(h)
        h = torch.cat([h, s3], dim = 1)     #拼接 为了将上采样的浓缩提炼信息和下采样时的粗糙原始信息融合
        h = self.up_attn3(self.up3_2(self.up3_1(h, t_emb), t_emb))


        h = self.up2(h)                                    # [128 32]
        h = torch.cat([h, s2], dim=1)                      # [256 32]
        h = self.up2_2(self.up2_1(h, t_emb), t_emb)

        h = self.up1(h)                                    # [64 64]
        h = torch.cat([h, s1], dim=1)                      # [128 64]
        h = self.up1_2(self.up1_1(h, t_emb), t_emb)

        return self.head(self.act(self.head_norm(h)))      # [B 3 64 64]


if __name__ == '__main__':
    device = 'cuda'
    temb = TimeEmbedding(base_ch=64).to(device)

    # 测试1：输出形状
    t = torch.randint(0, 1000, (4,), device=device)   # 随机 4 个时间步
    out = temb(t)
    print("输出形状:", out.shape)                      # 期待 torch.Size([4, 256])

    # 测试2：编码的性质 —— 相邻 t 指纹相近，远离 t 指纹差得远
    t1 = temb(torch.tensor([500], device=device))
    t2 = temb(torch.tensor([501], device=device))
    t3 = temb(torch.tensor([900], device=device))
    print("t=500 vs t=501 的距离:", (t1 - t2).norm().item())   # 应该小
    print("t=500 vs t=900 的距离:", (t1 - t3).norm().item())   # 应该大


    # ---- 2.2 ResBlock 测试 ----
    print("\n--- ResBlock 测试 ---")
    rb = ResBlock(in_ch=64, out_ch=128).to(device)
    x = torch.randn(2, 64, 64, 64, device=device)
    t_emb = temb(torch.tensor([100, 999], device=device))  # 复用上面的TimeEmbedding
    print("变通道输出:", rb(x, t_emb).shape)               # 期待 [2, 128, 64, 64]

    rb2 = ResBlock(in_ch=64, out_ch=64).to(device)
    print("同通道输出:", rb2(x, t_emb).shape)              # 期待 [2, 64, 64, 64]


    # ---- 2.3 SelfAttention 测试 ----
    print("\n--- SelfAttention 测试 ---")
    att = SelfAttention(ch=256).to(device)
    x = torch.randn(2, 256, 16, 16, device=device)
    print("Attention输出:", att(x).shape)          # 期待 [2, 256, 16, 16]

    # 验证零初始化:训练开始时注意力输出应与输入完全一样
    print("初始时是否等于恒等:", torch.allclose(att(x), x))



    # ---- 2.4 UNet 整体测试 ----
    print("\n--- UNet 测试 ---")
    unet = UNet(base_ch=64).to(device)
    n = sum(p.numel() for p in unet.parameters())
    print(f"参数量: {n/1e6:.1f}M")

    x = torch.randn(2, 3, 64, 64, device=device)
    t = torch.tensor([100, 999], device=device)
    with torch.no_grad():
        print("UNet输出:", unet(x, t).shape)   # 期待 [2, 3, 64, 64]

    # 显存压力测试: batch=8 完整前向+反传
    x = torch.randn(8, 3, 64, 64, device=device)
    t = torch.randint(0, 1000, (8,), device=device)
    y = unet(x, t)
    y.mean().backward()
    print(f"batch=8 前向+反传峰值显存: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")