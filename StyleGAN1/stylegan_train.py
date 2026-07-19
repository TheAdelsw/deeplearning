import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import cv2
import os
from torch.utils.data import DataLoader, Dataset
from StyleGAN import save_model, load_model


from StyleGAN import StyleGAN


class ImageDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.images = []

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        self.img_names = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(valid_ext)
        ]


        for name in self.img_names:
            img_path = os.path.join(self.img_dir,name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) #形状为H*W*C
        
            img = torch.from_numpy(img).permute(2,0,1).float() #转为C*H*W
            img = (img / 255.0) * 2 - 1 #归一化到0-1之间
            self.images.append(img)


    def __len__(self):
        return len(self.img_names)
    def __getitem__(self,idx):
        #name= self.img_names[idx]
        return self.images[idx]









def train(model, epochs, dataloader, save_path, Phase, Alpha):
    cnt = 0
    for epoch in range(epochs):
        for real_img in dataloader:
            cnt += 1
            real_img = real_img.to(model.device)

            # 判别器训练
            for _ in range(1):
                loss_C = model.TrainCell_C(real_img, Phase, Alpha)

            # 生成器训练
            z = torch.randn(real_img.size(0), 512, device=model.device)
            loss_G = model.TrainCell_G(z, Phase, Alpha)

            if cnt % 30 == 0:
                print(f"Phase[{Phase}] Alpha[{Alpha}] Iter [{cnt}] loss_C: {loss_C:.4f}  loss_G: {loss_G:.4f}")

            if cnt % 500 == 0:
                save_model(model, save_path)

    save_model(model, save_path)
    save_model(model, f"D:\Project\deeplearning\StyleGAN\model\SG_P{Phase}_A{Alpha}")




def make_photo(Phase, model, device, idx = 1):
    with torch.no_grad():
        z = -torch.rand(1, 512, device = device)*1.5

        w = model.mapping.Forward(z)
        img = model.Synthesis(w, Phase, 1.0)
        img = (img + 1) / 2
        # img = img.clamp(0, 1)      # [1, 3, 512, 512]

        # 2. [C,H,W] → [H,W,C]
        img = img[0].permute(1, 2, 0).cpu().numpy()     # [512, 512, 3], RGB

        # 3. RGB → BGR（因为 cv2.imshow 要 BGR）
        #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img * 255).clip(0, 255).astype('uint8')        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rf"D:\Project\deeplearning\StyleGAN\output\gen_{idx}.png", img)

        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()





if __name__ == "__main__":
    
    device = 'cuda'
    img_path = r"D:\source_data\ImageDataset\simple"
    model_path = r"D:\Project\deeplearning\StyleGAN\model\SG_1"
    model = StyleGAN(lr = 0.0004, device = device, use_amp = False)

    #cuDNN自动选择最优卷积算法
    torch.backends.cudnn.benchmark = True

    if os.path.exists(model_path):
        load_model(model, model_path)
        print("成功加载模型,继续训练")
    else:
        print("未找到模型,开始训练新模型")

    for i in range(1000):
        make_photo(7, model=model, device=device, idx=i)
    # make_photo(7, model=model, device=device)
    exit()



    print("加载数据集中......")
    dataset = ImageDataset(img_path)
    dataloader = DataLoader(
            dataset, 
            batch_size = 16, 
            shuffle = True, 
            num_workers = 0, 
            pin_memory = True, 
            #persistent_workers = True,
            #prefetch_factor = 0
            )
    print("数据集加载完成")


    try:
        
        # train(model = model, epochs=50, Phase = 7, Alpha = 0.0, dataloader = dataloader, save_path = model_path)
        # train(model = model, epochs=50, Phase = 7, Alpha = 0.25, dataloader = dataloader, save_path = model_path)
        # train(model = model, epochs=50, Phase = 7, Alpha = 0.5, dataloader = dataloader, save_path = model_path)
        train(model = model, epochs=50, Phase = 7, Alpha = 0.75, dataloader = dataloader, save_path = model_path)
        # train(model = model, epochs=100, Phase = 7, Alpha = 1.0, dataloader = dataloader, save_path = model_path)
        



    except KeyboardInterrupt:
        print("\n中断训练,正在保存模型...")
        save_model(model, model_path)
        print("已保存，退出。")
