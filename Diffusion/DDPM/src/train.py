#训练脚本 数据集 DDPM EMA AMP 梯度裁剪 checkpoint

import os
import cv2
import torch
import torchvision.utils as vutils

from torch.utils.data import DataLoader, Dataset

from scheduler import Scheduler
from unet import UNet
from ddpm import DDPM, EMA


#数据集
class ImageDataset(Dataset):
    def __init__(self, img_dir, img_size = 64):
        self.images = []
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        for name in os.listdir(img_dir):
            if not name.lower().endswith(valid_ext):    
                continue
            img = cv2.imread(os.path.join(img_dir, name))

            if img is None:
                continue

            if img.shape[0] != 64 or img.shape[1] != 64:
                img = cv2.resize(img, (img_size, img_size))     #[64, 64, 3] BGR
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = (img / 255) * 2 - 1

            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx] 


def save_checkpoint(path, unet, optimizer, ema, scaler, cnt):
    """state_dict: PyTorch通用的"参数名->张量"字典, 五样东西一次存齐"""
    torch.save({
        'unet':unet.state_dict(),
        'optimizer':optimizer.state_dict(),
        'ema_shadow':ema.shadow,
        'scaler':scaler.state_dict(),
        'cnt':cnt,
    }, path)



# 加载权重
def load_checkpoint(path, unet, optimizer, ema, scaler):
    ckpt = torch.load(path)
    unet.load_state_dict(ckpt['unet'])
    optimizer.load_state_dict(ckpt['optimizer'])
    #EMA 拷贝影子权重
    ema.shadow = ckpt['ema_shadow']
    scaler.load_state_dict(ckpt['scaler'])

    return ckpt['cnt']  #这边返回cnt又是代表什么


def train(ddpm, unet, ema, optimizer, scaler, dataloader, device,
        save_path, use_amp = True, epochs = 100, start_cnt = 0):
    cnt = start_cnt
    torch.backends.cudnn.benchmark = True   #卷积加速


    for epoch in range(epochs):
        for real_img in dataloader:
            cnt += 1
            real_img = real_img.to(device)

            #训练 AMP顺序固定
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = use_amp):
                loss = ddpm.p_losses(real_img)        # 用当前权重猜噪声
            scaler.scale(loss).backward()             # 放大loss再反传,防fp16下溢
            scaler.unscale_(optimizer)                # 还原真实梯度
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)  # 裁剪
            scaler.step(optimizer)                    # 更新权重 溢出时自动跳过本step
            scaler.update()                           # 调整放大倍数

            ema.update(unet)    #影子权重向step之后的最新权重缓慢平滑更新

            # 日志
            if cnt % 30 == 0:
                print(f"Iter [{cnt}] loss: {loss.item():.4f}")

            #存档
            if cnt % 300 == 0:
                save_checkpoint(save_path, unet, optimizer, ema, scaler, cnt)

            if cnt % 5000 == 0:
                ema.apply_shadow(unet)                # 影子权重
                imgs = ddpm.sample(16)                # 用影子生成
                ema.restore(unet)                     # 本体回归
                grid = vutils.make_grid((imgs + 1) / 2, nrow=4)
                vutils.save_image(grid, os.path.join(r'D:\Project\deeplearning\Diffusion\DDPM\output\preview', f"preview_{cnt}.png"))
                print(f"已保存预览图 preview_{cnt}.png")


    #训练结束 返回训练的次数
    return cnt


if __name__ == '__main__':
    device = 'cuda'
    img_path = r'D:\source_data\anime-face'
    pre_model = r'D:\Project\deeplearning\Diffusion\DDPM\model\ddpm_1.pt'

    save_path = r'D:\Project\deeplearning\Diffusion\DDPM\model\ddpm_1.pt'
    use_amp = True
    batch_size = 16
    lr = 2e-4
    epochs = 100

    sch = Scheduler(timesteps = 1000, device = device)
    unet = UNet(base_ch = 64).to(device)    #这里的.to(device)是把这个类的实例放到显存上吗
    ddpm = DDPM(unet, sch, device)
    ema = EMA(unet, decay = 0.999)

    #优化器 与 标准化器
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    #载入模型 
    if os.path.exists(pre_model):
        cnt = load_checkpoint(pre_model, unet, optimizer, ema, scaler)
        print(f"成功加载模型, 从第 {cnt} 步继续训练")
    else:
        cnt = 0
        print("未找到模型, 开始训练新模型")

    #加载数据
    print("加载数据集中......")
    dataset = ImageDataset(img_path, img_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)     
    print(f"数据集加载完成, 共 {len(dataset)} 张图")

    cnt = train(ddpm, unet, ema, optimizer, scaler, dataloader, device, save_path,
     epochs = epochs, start_cnt = cnt, use_amp = True)

    print(f"训练循环退出, 当前全局步数: {cnt}")