import torch






class NoiseInjection:
    def __init__(self, C, device):
        #一张图片中 每个通道的噪声注入强度不同
        self.strength = torch.zeros(C, requires_grad = True, device = device)

    def Forward(self, x):
        B, C, H, W = x.shape
        #一张图片中 每个像素噪声不同 通道仅有一个
        noise = torch.randn(B, 1, H, W, device = x.device)
        """
        经过广播后相乘 noise变成有C个通道 相同位置的像素在每个通道的噪声相同
        乘以不同的噪声强度 最终不同通道不同像素的噪声都不同
        不同张图片的同个通道的所有像素都是一样的噪声强度
        """
        out = x + noise * self.strength.view(1, C, 1, 1)
        return out

    def Params(self):
        return [self.strength]



