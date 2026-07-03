import torch
import torch.nn.functional as F
import torch.optim as optim


#本地文件
from mapping import MappingNet
from NoiseInjection import NoiseInjection
from AdaIN import AdaIN


class StyleGAN:
    def __init__(self, lr, device):
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

        self.opt_G = optim.Adam(self.Params_G(), lr = self.lr, betas = (0.0, 0.99))
        self.opt_C = optim.Adam(self.Params_C(), lr = self.lr, betas = (0.0, 0.99))

    def init_weights(self):
        pass


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

    def Critic(self, img):
        x = img
        for i in range(8):
            x = F.conv2d(x, self.critic_convs[i], stride = 1, padding = 1)
            x = F.leaky_relu(x, 0.2)
            #池化
            if i < 7:
                x = F.avg_pool2d(x, 2)
            #i == 8时此时x为 [B, 512, 4, 4]
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

        
