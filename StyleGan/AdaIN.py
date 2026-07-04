import torch
import torch.nn.functional as F



from Linear import Linear

#IN + w生成的scale和bias
class AdaIN:
    def __init__(self, C, w_dim = 512, device = 'cuda'):
        self.C = C
        
        #生成风格参数的全连接层
        self.Affine = Linear(w_dim, C * 2, device)
        with torch.no_grad():
            self.Affine.W.zero_()
            self.Affine.b.zero_()
            self.Affine.b[:self.C].fill_(1.0)

    #风格注入 给定x 做IN归一 然后风格参数注入
    def Forward(self, x, w):
        B, C, H, W = x.shape
        assert C == self.C 

        x = F.instance_norm(x, eps = 1e-8)

        style = self.Affine.Forward(w)#噪声8层全连接生成的w
        scale, bias = style[:, :C], style[:, C:]    #batch, C
        scale = scale.view(B, C, 1, 1)
        bias = bias.view(B, C, 1, 1)
        
        out = scale * x + bias
        return out

    def Params(self):
        return self.Affine.Params()


