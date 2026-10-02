import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Hid(nn.Module):
    def __init__(self, ch):
        super(Hid, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class GradientS(nn.Module):
    def __init__(self, hid_num, image_channel, channel_expand):
        super(GradientS, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channel,
                      out_channels=channel_expand,
                      kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.hid = nn.ModuleList([Hid(channel_expand) for _ in range(hid_num)])

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=channel_expand,
                      out_channels=image_channel,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(num_features=image_channel)
        )


    def forward(self, x):
        x = self.encoder(x)
        for net in self.hid: x = net(x)
        # 这里S的梯度必须可正可负
        # 否则OEM中O<=F，模型无法正常训练
        return self.decoder(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio = 4):
        super(SEBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, 1, 0),
            nn.BatchNorm2d(num_features=channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.attn(x)
    
class DCPM(nn.Module):
    def __init__(self, in_c, hid_c, reduction_ratio = 4):
        super(DCPM, self).__init__()
        self.encoder = nn.Conv2d(in_channels=in_c, out_channels=hid_c, kernel_size=3, padding=1)

        self.attn = SEBlock(channels=hid_c, reduction_ratio=reduction_ratio)
        self.conv = nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1)

        self.decoder = nn.Conv2d(in_channels=hid_c, out_channels=in_c, kernel_size=1)

    def forward(self, x):
        x_enc = F.relu(self.encoder(x))

        P_origin = self.conv(x_enc)

        kernel_w = self.conv.weight.sum(1)[:, None, :, :]
        kernel_sum = kernel_w

        Ps = F.conv2d(input=x_enc, weight=kernel_sum, stride=self.conv.stride, padding=self.conv.padding, groups=x_enc.shape[1])

        attn_weight = self.attn(x_enc)

        return self.decoder(attn_weight * Ps - P_origin)
    
class OEM(nn.Module):
    def __init__(self, image_channel, channel_expand, lo):
        super(OEM, self).__init__()
        self.dcpm = DCPM(in_c=image_channel, hid_c=channel_expand, reduction_ratio=4)
        self.grad_S = GradientS(hid_num=lo,
                                image_channel=image_channel,
                                channel_expand=channel_expand)

        # 这里官方实现初值为0.01，尝试以1为初值会崩
        self.rho = nn.Parameter(torch.tensor(0.01))

    def forward(self, D, O, B):
        Fussion = D + O - B
        grad = self.rho * self.grad_S(Fussion + self.dcpm(Fussion))

        return Fussion - grad
