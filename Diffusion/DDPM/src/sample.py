#加载训练时不断更新的影子权重 批量生成图片

import os
import torch
import torchvision.utils as vutils


from scheduler import Scheduler
from unet import UNet
from ddpm import DDPM


if __name__ == '__main__':
    device = 'cuda'
    model_path = r'D:\Project\deeplearning\Diffusion\DDPM\model\ddpm_1.pt'
    output_dir = r'D:\Project\deeplearning\Diffusion\DDPM\output\gen'
    
    n = 64

    sch = Scheduler(timesteps = 1000, device = device)
    unet = UNet(base_ch = 64).to(device)   
    ddpm = DDPM(unet, sch, device)

    #加载模型   map_location 存档是保存在gpu上的
    ckpt = torch.load(model_path, map_location = device)
    print(f"存档记录 第{ckpt['cnt']} 步")
    unet.load_state_dict(ckpt['unet'])  #这一步可以省略 直接加载影子权重

    unet.load_state_dict(ckpt['ema_shadow'])
    unet.eval()     #dropout在卷积网络中是怎样体现随机失活神经元的 明明不像全连接

    with torch.no_grad():
        imgs = ddpm.sample(n)

    #反向过程有误差和自动附加的噪声 可能会导致值超出范围
    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1) #裁剪到[0, 1]
    imgs = imgs.flip(1) #将第1维 3通道翻转 第0维是批次
    

    grid = vutils.make_grid(imgs, nrow=8)   #torchvision工具函数 每行八张
    out_file = os.path.join(output_dir, f"gen_{ckpt['cnt']}.png")

    vutils.save_image(grid, out_file)
    print(f"已保存 {n} 张生成图: {out_file}")
