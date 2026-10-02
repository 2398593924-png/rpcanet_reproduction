import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BackgroundApp import BAM
from model.ImageRes import IRM
from model.ObjectExtractor import OEM
from configs import config

class OneStage(nn.Module):
    def __init__(self, in_c, hid_c, lo, ld):
        super(OneStage, self).__init__()
        self.oem = OEM(in_c, hid_c, lo)
        self.bam = BAM(in_c, hid_c)
        self.irm = IRM(in_c, hid_c, ld)

    def forward(self, Dk, Ok, Bh, Bc):
        Bk_new, Bh_new, Bc_new = self.bam(Dk, Ok, Bh, Bc)
        Ok_new = self.oem(Dk, Ok, Bk_new)
        Dk_new = self.irm(Bk_new, Ok_new)
        
        return Dk_new, Ok_new, Bh_new, Bc_new

class RPCANet(nn.Module):
    def __init__(self, in_c, hid_c, lo, ld, stage_num):
        super(RPCANet, self).__init__()
        self.net = nn.ModuleList([OneStage(in_c, hid_c, lo, ld) for _ in range(stage_num)])
        self.hid_c = hid_c

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        Dk, Ok = x, torch.zeros_like(x)
        Bh = torch.zeros(batch_size, self.hid_c, x.shape[2], x.shape[3]).to(config.device)
        Bc = torch.zeros(batch_size, self.hid_c, x.shape[2], x.shape[3]).to(config.device)

        for stage in self.net:
            Dk, Ok, Bh, Bc = stage(Dk, Ok, Bh, Bc)

        return Dk, Ok