import torch


class Linear:
    def __init__(self, in_feature, out_feature, device):
        self.W = torch.randn(in_feature, out_feature, requires_grad = True, device =  device)
        self.b = torch.zeros(out_feature, requires_grad = True, device = device)
        self.W.data *= 0.02
    
    def Forward(self, x):
        #x以in行行向量 W为in*out矩阵 输出out维
        out = x @ self.W + self.b
        return out

    def Params(self):
        return [self.W , self.b]
