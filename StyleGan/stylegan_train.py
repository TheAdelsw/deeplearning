import torch
import cv2
import os
from torch.utils.data import DataLoader, Dataset



from StyleGAN import StyleGAN


class ImageDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self,idx):
        name= self.img_names[idx]
        img_path = os.path.join(self.img_dir,name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) #形状为H*W*C
        # if not img.shape[0]==512:
        #     img = cv2.resize(img, (512, 512))
        
        img = torch.from_numpy(img).permute(2,0,1).float() #转为C*H*W
        img = (img / 255.0) * 2 - 1 #归一化到0-1之间
        return img


def save_model(model, path):
    state = {}

    # ===== Mapping Network (8 个 Linear) =====
    fcs = [model.mapping.fc1, model.mapping.fc2, model.mapping.fc3, model.mapping.fc4,
           model.mapping.fc5, model.mapping.fc6, model.mapping.fc7, model.mapping.fc8]
    for i, fc in enumerate(fcs):
        state[f'map_fc{i}_W'] = fc.W.data.cpu()
        state[f'map_fc{i}_b'] = fc.b.data.cpu()

    # ===== const_input =====
    state['const_input'] = model.const_input.data.cpu()

    # ===== Synthesis Network (8 stages) =====
    for i in range(8):
        state[f'syn_conv1_{i}'] = model.syn_convs[i][0].data.cpu()
        state[f'syn_conv2_{i}'] = model.syn_convs[i][1].data.cpu()
        state[f'syn_torgb_{i}'] = model.syn_torgb[i].data.cpu()
        # Noise strength
        state[f'syn_ns1_{i}'] = model.syn_noises[i][0].strength.data.cpu()
        state[f'syn_ns2_{i}'] = model.syn_noises[i][1].strength.data.cpu()
        # AdaIN Affine (Linear 的 W 和 b)
        state[f'syn_ada1_W_{i}'] = model.syn_adains[i][0].Affine.W.data.cpu()
        state[f'syn_ada1_b_{i}'] = model.syn_adains[i][0].Affine.b.data.cpu()
        state[f'syn_ada2_W_{i}'] = model.syn_adains[i][1].Affine.W.data.cpu()
        state[f'syn_ada2_b_{i}'] = model.syn_adains[i][1].Affine.b.data.cpu()

    # ===== Critic (8 conv + 1 final) =====
    for i in range(8):
        state[f'critic_conv_{i}'] = model.critic_convs[i].data.cpu()
    state['critic_final'] = model.critic_final_conv.data.cpu()

    # ===== Optimizers =====
    state['opt_G'] = model.opt_G.state_dict()
    state['opt_C'] = model.opt_C.state_dict()

    torch.save(state, path)
    print(f"模型已保存到 {path}")
    


def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu')

    # ===== Mapping Network =====
    fcs = [model.mapping.fc1, model.mapping.fc2, model.mapping.fc3, model.mapping.fc4,
           model.mapping.fc5, model.mapping.fc6, model.mapping.fc7, model.mapping.fc8]
    for i, fc in enumerate(fcs):
        fc.W.data.copy_(checkpoint[f'map_fc{i}_W'].to(model.device))
        fc.b.data.copy_(checkpoint[f'map_fc{i}_b'].to(model.device))

    # ===== const_input =====
    model.const_input.data.copy_(checkpoint['const_input'].to(model.device))

    # ===== Synthesis Network =====
    for i in range(8):
        model.syn_convs[i][0].data.copy_(checkpoint[f'syn_conv1_{i}'].to(model.device))
        model.syn_convs[i][1].data.copy_(checkpoint[f'syn_conv2_{i}'].to(model.device))
        model.syn_torgb[i].data.copy_(checkpoint[f'syn_torgb_{i}'].to(model.device))
        model.syn_noises[i][0].strength.data.copy_(checkpoint[f'syn_ns1_{i}'].to(model.device))
        model.syn_noises[i][1].strength.data.copy_(checkpoint[f'syn_ns2_{i}'].to(model.device))
        model.syn_adains[i][0].Affine.W.data.copy_(checkpoint[f'syn_ada1_W_{i}'].to(model.device))
        model.syn_adains[i][0].Affine.b.data.copy_(checkpoint[f'syn_ada1_b_{i}'].to(model.device))
        model.syn_adains[i][1].Affine.W.data.copy_(checkpoint[f'syn_ada2_W_{i}'].to(model.device))
        model.syn_adains[i][1].Affine.b.data.copy_(checkpoint[f'syn_ada2_b_{i}'].to(model.device))

    # ===== Critic =====
    for i in range(8):
        model.critic_convs[i].data.copy_(checkpoint[f'critic_conv_{i}'].to(model.device))
    model.critic_final_conv.data.copy_(checkpoint['critic_final'].to(model.device))

    # ===== Optimizers =====
    model.opt_G.load_state_dict(checkpoint['opt_G'])
    model.opt_C.load_state_dict(checkpoint['opt_C'])

    # 把优化器内部状态搬到正确 device
    for state in model.opt_G.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(model.device)
    for state in model.opt_C.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(model.device)

    print(f"模型已从 {path} 加载")






def train(model, epochs, dataloader, save_path):
    cnt = 0
    for epoch in range(epochs):
        for real_img in dataloader:
            cnt += 1
            real_img = real_img.to(model.device)

            #判别器训练
            for _ in range(3):
                loss_C = model.TrainCell_C(real_img)

            #生成器训练
            z = torch.randn(real_img.size(0), 512, device=model.device)
            loss_G = model.TrainCell_G(z)

            if cnt % 30 == 0:
                print(f"Iter [{cnt}] loss_C: {loss_C:.4f}  loss_G: {loss_G:.4f}")
            
            if cnt % 500 == 0:
                save_model(model, save_path)

    save_model(model, save_path)








if __name__ == "__main__":
    
    device = 'cuda'
    img_path = r"D:\source_data\ImageDataset\simple"
    model_path = r"D:\Project\deeplearning\StyleGan\model\stylengan_v1"
    model = StyleGAN(lr = 0.001, device = device)


    dataset = ImageDataset(img_path)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

    if os.path.exists(model_path):
        load_model(model, model_path)
        print("成功加载模型,继续训练")
    else:
        print("未找到模型,开始训练新模型")

    try:
        train(model = model, epochs=100, dataloader = dataloader, save_path = model_path)

    except KeyboardInterrupt:
        print("\n中断训练,正在保存模型...")
        save_model(model, model_path)
        print("已保存，退出。")
