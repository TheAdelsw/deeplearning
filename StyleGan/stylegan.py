import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim


#本地文件
from mapping import MappingNet
from NoiseInjection import NoiseInjection
from AdaIN import AdaIN


class StyleGAN:
    def __init__(self, lr, device, use_amp):
        self.use_amp = use_amp
        self.gp_lambda = 8
        self.lr = lr
        self.device = device
        self.mapping = MappingNet(z_dim = 512, w_dim = 512, device = device)

        #分辨率
        self.resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
        #通道数
        self.ch_in = [512, 512, 512, 512, 512, 256, 128, 64]
        self.ch_out = [512, 512, 512, 512, 256, 128, 64, 32]

        #初始画布
        self.const_input = torch.randn(1, self.ch_in[0], 4, 4, requires_grad = True, device = device)

        self.syn_convs = []
        self.syn_noises = []
        self.syn_adains = []
        self.syn_torgb = []
        #Noise → Conv → LeakyReLU → AdaIN
        for i in range(8):
            cin = self.ch_in[i]
            cout = self.ch_out[i]

            noise1 = NoiseInjection(cin, device)
            noise2 = NoiseInjection(cout, device)
            conv1_w = torch.randn(cout, cin, 3, 3, requires_grad = True, device = device)
            conv2_w = torch.randn(cout, cout, 3, 3, requires_grad = True, device = device)
            adain1 = AdaIN(cout, w_dim = 512, device = device)
            adain2 = AdaIN(cout, w_dim = 512, device = device)
            torgb_w = torch.randn(3, cout, 1, 1, requires_grad = True, device = device)
            self.syn_convs.append([conv1_w, conv2_w])
            self.syn_torgb.append(torgb_w)
            self.syn_noises.append([noise1, noise2])
            self.syn_adains.append([adain1, adain2])
        #                   512 256 128 64   32   16   8    4 
        self.critic_ch_in = [3, 32, 64, 128, 256, 512, 512, 512]
        self.critic_ch_out = [32, 64, 128, 256, 512, 512, 512, 512]

        self.critic_convs = []
        self.critic_fromrgb = []
        
        self.critic_final_conv = torch.randn(1, self.critic_ch_out[7], 4, 4, requires_grad = True, device = device)
        #为了训练速度提高 使用3*3的卷积核 然后池化缩放尺寸
        for i in range(8):
            cin = self.critic_ch_in[i]
            cout = self.critic_ch_out[i]
            conv_w = torch.randn(cout, cin, 3, 3, requires_grad = True, device = device)
            self.critic_convs.append(conv_w)
            conv_fromrgb = torch.randn(self.critic_ch_in[i], 3, 1, 1, requires_grad=True, device=device)
            self.critic_fromrgb.append(conv_fromrgb)

        #必须先初始化权重 然后创建优化器
        self.init_weights()

        self.opt_G = optim.Adam(self.Params_G(), lr = self.lr, betas = (0.0, 0.99))
        self.opt_C = optim.Adam(self.Params_C(), lr = self.lr, betas = (0.0, 0.99))

        #AMP混合精度
        self.scaler = torch.cuda.amp.GradScaler()


    def init_weights(self):
        #生成器卷积
        for conv in self.syn_convs:
            for w in conv:
                torch.nn.init.kaiming_normal_(w, mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0.2)


        #toRGB
        for w in self.syn_torgb:
            w.data.normal_(0, 0.01)


        #判别器
        for i in range(8):
            torch.nn.init.kaiming_normal_(self.critic_convs[i], mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0.2)
            torch.nn.init.kaiming_normal_(self.critic_fromrgb[i], mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0.2)

        torch.nn.init.kaiming_normal_(self.critic_final_conv, mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0.2)
        self.critic_final_conv.data *= 0.01

        


    def Synthesis(self, W, Phase, Alpha):
        #phase 训练到第phase个stage
        #alpha 过渡参数
        B = W.shape[0]

        x = self.const_input.repeat(B, 1, 1, 1) # [B, 512, 4, 4]
        rgb = None 

        #8个stage
        for i in range(Phase):
            #上采样
            if i > 0 :
                x = F.interpolate(x, scale_factor = 2, mode='bilinear', align_corners = False)

            #Block1
            x = self.syn_noises[i][0].Forward(x)
            x = F.conv2d(x, self.syn_convs[i][0], stride = 1, padding = 1)
            x = F.leaky_relu(x, 0.2)
            x = self.syn_adains[i][0].Forward(x, W)

            #Block2
            x = self.syn_noises[i][1].Forward(x)
            x = F.conv2d(x, self.syn_convs[i][1], stride = 1, padding = 1)
            x = F.leaky_relu(x, 0.2)
            x = self.syn_adains[i][1].Forward(x, W)

            rgb_new = F.conv2d(x, self.syn_torgb[i], stride = 1, padding = 0)

            if rgb == None:
                rgb = rgb_new
            elif   i == Phase - 1 and Alpha < 1.0:
                rgb = Alpha * rgb_new + (1 - Alpha) * F.interpolate(rgb, scale_factor = 2, mode = 'bilinear', align_corners = False)
            else:
                rgb = rgb_new + F.interpolate(rgb, scale_factor = 2, mode = 'bilinear', align_corners = False)


        return torch.tanh(rgb)

    def Generate(self, z, Phase = 8, Alpha = 1.0):
        with torch.no_grad():
            w = self.mapping.Forward(z)
            img = self.Synthesis(w, Phase, Alpha)
        return img 

    def Critic(self, img, Phase, Alpha):
        start = 8  - Phase

        x = F.conv2d(img, self.critic_fromrgb[start], stride = 1, padding = 0)

        for i in range(start, 8):
            if i == start and Phase > 1 and Alpha < 1.0:
                #判别器的渐进训练
                x_new = F.conv2d(x, self.critic_convs[i], stride = 1, padding = 1)
                x_new = F.leaky_relu(x_new, 0.2)
                #池化
                if i < 7:
                    #i == 7时此时x为 [B, 512, 4, 4]
                    x_new = F.avg_pool2d(x_new, 2)
                x_old = F.avg_pool2d(img, 2)
                x_old = F.conv2d(x_old, self.critic_fromrgb[i+1], stride = 1, padding = 0)
                x = Alpha * x_new + (1 - Alpha) * x_old

            else:
                x = F.conv2d(x, self.critic_convs[i], stride=1, padding=1)
                x = F.leaky_relu(x, 0.2)
                if i < 7:
                    x = F.avg_pool2d(x, 2)

        out = F.conv2d(x, self.critic_final_conv, stride = 1, padding = 0)
        return out.view(-1)


    def Params_G(self):
        params = []
        #mapping 
        params += self.mapping.Params()
        #画布
        params.append(self.const_input)

        for i in range(8):
            params += self.syn_noises[i][0].Params()
            params += self.syn_noises[i][1].Params()
            params.append(self.syn_convs[i][0])
            params.append(self.syn_convs[i][1])
            params += self.syn_adains[i][0].Params()
            params += self.syn_adains[i][1].Params()
            params.append(self.syn_torgb[i])

        return params

    def Params_C(self):
        params = []
        for i in range(8):
            params.append(self.critic_convs[i])
            params.append(self.critic_fromrgb[i])

        params.append(self.critic_final_conv)
        return params

        

    def gradient_penalty(self, real_img, fake_img, Phase, Alpha):
        alpha = torch.rand(real_img.size(0), 1, 1, 1,device = self.device)
        interpolates = alpha * real_img + (1-alpha) * fake_img
        interpolates.requires_grad_(True)

        d_interpolates = self.Critic(interpolates, Phase, Alpha)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),#自动匹配 device
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    

    def TrainCell_C(self, real_img, Phase, Alpha):
        target_size = self.resolutions[Phase - 1]
        real_img = F.interpolate(real_img, size=target_size, mode='bilinear', align_corners=False)

        B = real_img.size(0)
        z = torch.randn(B, 512, device = self.device)

        if self.use_amp :
            #混合精度 autocast
            with torch.amp.autocast('cuda'):
                w = self.mapping.Forward(z)
                fake_img = self.Synthesis(w, Phase, Alpha).detach()

                real_score = self.Critic(real_img, Phase, Alpha)
                fake_score = self.Critic(fake_img, Phase, Alpha)

            gp = self.gradient_penalty(real_img, fake_img, Phase, Alpha)

            loss_C = fake_score.float().mean() - real_score.float().mean() + self.gp_lambda * gp

            self.opt_C.zero_grad()
            # loss_C.backward()
            self.scaler.scale(loss_C).backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.Params_C(), max_norm=10.0)

            # self.opt_C.step()
            self.scaler.step(self.opt_C)
            self.scaler.update()
        else:
            w = self.mapping.Forward(z)
            fake_img = self.Synthesis(w, Phase, Alpha).detach()

            real_score = self.Critic(real_img, Phase, Alpha)
            fake_score = self.Critic(fake_img, Phase, Alpha)
            gp = self.gradient_penalty(real_img, fake_img, Phase, Alpha)

            loss_C = fake_score.float().mean() - real_score.float().mean() + self.gp_lambda * gp

            self.opt_C.zero_grad()
            loss_C.backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.Params_C(), max_norm=10.0)

            self.opt_C.step()



        #   用以监测训练效果
        if torch.rand(1).item() < 0.01:  # 1% 概率打印
            print(f"  [C] real:{real_score.mean():.2f} fake:{fake_score.mean():.2f} gap:{(real_score.mean()-fake_score.mean()):.2f} gp:{gp:.4f}")


        return loss_C.item()



    def TrainCell_G(self, z, Phase, Alpha):#生成时外部提供风格源噪声z
        
        if self.use_amp :
            #AMP混合精度
            with torch.amp.autocast('cuda'):
                w = self.mapping.Forward(z)
                fake_img = self.Synthesis(w, Phase, Alpha)

                fake_score = self.Critic(fake_img, Phase, Alpha)
                loss_G = -fake_score.mean()

            self.opt_G.zero_grad()
            # loss_G.backward()
            self.scaler.scale(loss_G).backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.Params_G(), max_norm=10.0)

            # self.opt_G.step()
            self.scaler.step(self.opt_G)
            self.scaler.update()
        else :
            w = self.mapping.Forward(z)
            fake_img = self.Synthesis(w, Phase, Alpha)

            fake_score = self.Critic(fake_img, Phase, Alpha)
            loss_G = -fake_score.mean()

            self.opt_G.zero_grad()
            loss_G.backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.Params_G(), max_norm=10.0)

            self.opt_G.step()

        return loss_G.item()












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
        state[f'critic_fromrgb_{i}'] = model.critic_fromrgb[i].data.cpu()
    state['critic_final'] = model.critic_final_conv.data.cpu()

    # ===== Optimizers =====
    state['opt_G'] = model.opt_G.state_dict()
    state['opt_C'] = model.opt_C.state_dict()

    #scaler
    state['scaler'] = model.scaler.state_dict()


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
        model.critic_fromrgb[i].data.copy_(checkpoint[f'critic_fromrgb_{i}'].to(model.device))
    model.critic_final_conv.data.copy_(checkpoint['critic_final'].to(model.device))

    # ===== Optimizers =====
    model.opt_G.load_state_dict(checkpoint['opt_G'])
    model.opt_C.load_state_dict(checkpoint['opt_C'])
     
    #scaler
    model.scaler.load_state_dict(checkpoint['scaler'])


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