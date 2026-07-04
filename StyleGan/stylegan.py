import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim


#本地文件
from mapping import MappingNet
from NoiseInjection import NoiseInjection
from AdaIN import AdaIN


class StyleGAN:
    def __init__(self, lr, device):
        self.gp_lambda = 10
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

        self.critic_ch_in = [3, 32, 64, 128, 256, 512, 512, 512]
        self.critic_ch_out = [32, 64, 128, 256, 512, 512, 512, 512]

        self.critic_convs = []
        self.critic_final_conv = torch.randn(1, self.critic_ch_out[7], 4, 4, requires_grad = True, device = device)
        #为了训练速度提高 使用3*3的卷积核 然后池化缩放尺寸
        for i in range(8):
            cin = self.critic_ch_in[i]
            cout = self.critic_ch_out[i]
            conv_w = torch.randn(cout, cin, 3, 3, requires_grad = True, device = device)
            self.critic_convs.append(conv_w)

        #必须先初始化权重 然后创建优化器
        self.init_weights()

        self.opt_G = optim.Adam(self.Params_G(), lr = self.lr, betas = (0.0, 0.99))
        self.opt_C = optim.Adam(self.Params_C(), lr = self.lr, betas = (0.0, 0.99))

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

        torch.nn.init.kaiming_normal_(self.critic_final_conv, mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0.2)
        self.critic_final_conv.data *= 0.01

        


    def Synthesis(self, W):
        B = W.shape[0]

        x = self.const_input.repeat(B, 1, 1, 1) # [B, 512, 4, 4]
        rgb = None 

        #8个stage
        for i in range(8):
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
            else:
                rgb = rgb_new + F.interpolate(rgb, scale_factor = 2, mode = 'bilinear', align_corners = False)


        return torch.tanh(rgb)

    def Generate(self, z):
        with torch.no_grad():
            w = self.mapping.Forward(z)
            img = self.Synthesis(w)
        return img 

    def Critic(self, img):
        x = img
        for i in range(8):
            x = F.conv2d(x, self.critic_convs[i], stride = 1, padding = 1)
            x = F.leaky_relu(x, 0.2)
            #池化
            if i < 7:
                x = F.avg_pool2d(x, 2)
            #i == 7时此时x为 [B, 512, 4, 4]
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

        params.append(self.critic_final_conv)
        return params

        

    def gradient_penalty(self, real_img, fake_img):
        alpha = torch.rand(real_img.size(0), 1, 1, 1,device = self.device)
        interpolates = alpha * real_img + (1-alpha) * fake_img
        interpolates.requires_grad_(True)

        d_interpolates = self.Critic(interpolates)

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
    

    def TrainCell_C(self, real_img):
        B = real_img.size(0)
        z = torch.randn(B, 512, device = self.device)
        w = self.mapping.Forward(z)
        fake_img = self.Synthesis(w).detach()

        real_score = self.Critic(real_img)
        fake_score = self.Critic(fake_img)

        gp = self.gradient_penalty(real_img, fake_img)

        loss_C = fake_score.mean() - real_score.mean() + self.gp_lambda * gp

        self.opt_C.zero_grad()
        loss_C.backward()
        self.opt_C.step()

        return loss_C.item()



    def TrainCell_G(self, z):#生成时外部提供风格源噪声z
        w = self.mapping.Forward(z)
        fake_img = self.Synthesis(w)

        fake_score = self.Critic(fake_img)
        loss_G = -fake_score.mean()

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        return loss_G.item()