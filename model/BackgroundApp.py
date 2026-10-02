import torch
import torch.nn as nn
import torch.nn.functional as F
 
class RB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):
        super(RB, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

class ConvC2C(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 3, padding = 1):
        super(ConvC2C, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = in_c,
                      out_channels = out_c,
                      kernel_size = kernel_size,
                      padding = padding
                      ),
            nn.BatchNorm2d(num_features = out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):
        super().__init__()
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=1)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=1)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=1)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=1)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=1)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=1)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=1)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=1)

    def forward(self, x, h, c):
        if h is None and c is None:
            h = torch.zeros_like(x)
            c = torch.zeros_like(x)
            
        f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h))
        i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h))
        o = torch.sigmoid(self.conv_xo(x) + self.conv_ho(h))
        j = torch.tanh(self.conv_xj(x) + self.conv_hj(h))
        c = f * c + i * j
        h = o * torch.tanh(c)
        return h, h, c
    
class BAM(nn.Module):
    def __init__(self, image_channel = 1, channel_expan = 32):
        super(BAM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channel,
                      out_channels=channel_expan,
                      kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=channel_expan),
            nn.ReLU(),
            RB(in_channels=channel_expan,
               out_channels=channel_expan,
               kernel_size=3)
        )

        self.decoder = nn.Sequential(
            RB(in_channels=channel_expan,
               out_channels=channel_expan,
               kernel_size=3),
               nn.Conv2d(in_channels=channel_expan,
                         out_channels=image_channel,
                         kernel_size=3,
                         padding=1),
                nn.BatchNorm2d(num_features=image_channel),
                nn.ReLU()
        )

        self.Memory = ConvLSTM(channel_expan, channel_expan, 3)

    def forward(self, D, O, Bh, Bc):
        Fusion_F = D - O
        Memory_Arg, Bh_new, Bc_new = self.Memory(self.encoder(Fusion_F), Bh, Bc)
        # 输出经过记忆增强的特征，以及新一轮的ConvLSTM参数
        return self.decoder(Memory_Arg) + Fusion_F, Bh_new, Bc_new
