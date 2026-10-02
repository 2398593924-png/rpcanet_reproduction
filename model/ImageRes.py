import torch
import torch.nn as nn
import torch.nn.functional as F

class MidNet(nn.Module):
    def __init__(self, ch):
        super(MidNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class IRM(nn.Module):
    def __init__(self, image_channel, channel_expand, ld):
        super(IRM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channel, channel_expand, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel_expand),
            nn.ReLU()
        )

        self.hid = nn.ModuleList([MidNet(channel_expand) for _ in range(ld)])

        # 原文: "identical convolution setting"
        self.conv2 = nn.Conv2d(channel_expand, image_channel, kernel_size=3, padding=1)

    def forward(self, Bk, Ok):
        x = self.conv1(Bk + Ok)
        for blk in self.hid: x = blk(x)
        
        return self.conv2(x)