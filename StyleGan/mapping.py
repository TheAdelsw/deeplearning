#包
import torch
import torch.nn.functional as F


#本地文件
from Linear import Linear


class MappingNet:
    def __init__(self, z_dim = 512, w_dim = 512, device = 'cuda'):
        self.fc1 = Linear(512, 512, device)
        self.fc2 = Linear(512, 512, device)
        self.fc3 = Linear(512, 512, device)
        self.fc4 = Linear(512, 512, device)
        self.fc5 = Linear(512, 512, device)
        self.fc6 = Linear(512, 512, device)
        self.fc7 = Linear(512, 512, device)
        self.fc8 = Linear(512, 512, device)
    
    def Forward(self, z):
        #噪声向量归一化 PixelNorm  z: batch,512
        z = z / torch.sqrt(torch.mean(z**2, dim = 1, keepdim = True) + 1e-8)
        out = z
        out = self.fc1.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc2.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc3.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc4.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc5.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc6.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc7.Forward(out); out = F.leaky_relu(out, 0.2)
        out = self.fc8.Forward(out) #最后一层不激活

        return out
    
    def Params(self):
        params = []
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, 
                   self.fc5, self.fc6, self.fc7, self.fc8]:
            params += fc.Params()
        return params


