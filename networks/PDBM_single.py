import torch
import torch.nn.functional as F
from torch import nn
from .Resnet50 import Resnet50

class PDBM_single(nn.Module):
    def __init__(self):
        super(PDBM_single, self).__init__()
        self.backbone = Resnet50()
        self.PDC_encoder = PDC(dilation_list=(2, 4, 8, 16))
        self.final_pre = nn.Conv2d(32, 1, 1)
        initialize_weights(self.PDC_encoder, self.final_pre)

    def forward(self, x):
        # x shape: B, C, W, H
        x_size = x.size()
        x = self.backbone(x)
        x = self.PDC_encoder(x)
        predict = self.final_pre(x)
        predict = F.interpolate(predict, size=x_size[-2:], mode='bilinear', align_corners=True)
        return predict

class PDC(nn.Module):
    def __init__(self, dilation_list=(2, 4, 8, 16)):
        super(PDC, self).__init__()
        self.p1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=dilation_list[0], dilation=dilation_list[0], bias=False),
                                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.p2 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=dilation_list[1], dilation=dilation_list[1], bias=False),
                                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.p3 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=dilation_list[2], dilation=dilation_list[2], bias=False),
                                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.p4 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=dilation_list[3], dilation=dilation_list[3], bias=False),
                                nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.reshape = nn.Sequential(nn.Conv2d(2048+512*4, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, x):
        # x shape: (B, C, H, W)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return self.reshape(torch.cat([x, p1, p2, p3, p4], dim=1))  # return shape: (B, C, H, W)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()