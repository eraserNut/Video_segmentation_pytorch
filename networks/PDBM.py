import torch
import torch.nn.functional as F
from torch import nn
from .Resnet50 import Resnet50
from .ConvLSTM import ConvLSTM
from config import PDBM_single_path

class PDBM(nn.Module):
    def __init__(self):
        super(PDBM, self).__init__()
        self.backbone = Resnet50()
        self.PDC_encoder = PDC(dilation_list=(2, 4, 8, 16))
        self.PDB_decoder = PDB(dilation_list=(1, 2))

        # load pretrained model from PDBM single
        model_dict = self.state_dict()
        pretrained_dict = torch.load(PDBM_single_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        # initialize scratch module
        initialize_weights(self.PDB_decoder)

    def forward(self, seq):
        # seq shape: B, T, C, W, H
        seq_size = seq.size()
        seq_f = []
        for t in range(seq_size[1]):
            img = seq[:, t, :, :, :]  # img shape: B, C, W, H
            img = self.backbone(img)
            img = self.PDC_encoder(img)
            seq_f.append(img)
        seq_f = torch.stack(seq_f, dim=1)  # seq feature shape:(B, T, C=32, W, H)
        predict_list = self.PDB_decoder(seq_f)
        predict_list_up = []
        for predict in predict_list:
            predict_list_up.append(F.interpolate(predict, size=seq_size[-2:], mode='bilinear', align_corners=True))
        return predict_list_up

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
        return self.reshape(torch.cat([x, p1, p2, p3, p4], dim=1))


class PDB(nn.Module):
    def __init__(self, dilation_list=(1, 2)):
        super(PDB, self).__init__()
        self.final_pre = nn.Conv2d(32*len(dilation_list), 1, 1)
        self.DB1 = DB_ConvLSTM(dilation_rate=1)
        self.DB2 = DB_ConvLSTM(dilation_rate=2)

    def forward(self, seq_f):
        pdb_f = torch.cat([self.DB1(seq_f), self.DB2(seq_f)], dim=2)
        predict_list = []
        for t in range(seq_f.size()[1]):
            predict_list.append(self.final_pre(pdb_f[:, t, :, :, :]))
        return predict_list


class DB_ConvLSTM(nn.Module):
    def __init__(self, dilation_rate=1):
        super(DB_ConvLSTM, self).__init__()
        self.ConvLSTM = ConvLSTM(input_dim=32,
                         hidden_dim=[32],
                         kernel_size=(3, 3),
                         num_layers=1,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False,
                         dilation_rate=dilation_rate)
        self.ConvLSTM_reverse = ConvLSTM(input_dim=32,
                         hidden_dim=[32],
                         kernel_size=(3, 3),
                         num_layers=1,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False,
                         dilation_rate=dilation_rate)

    def forward(self, seq_f):
        seq_size = seq_f.size()
        # f1->f2->f3
        layer_output_list, _ = self.ConvLSTM(seq_f)
        seq_f_lstm1 = layer_output_list[0]  # num_layer=1
        # reverse seq
        reverse_seq_f_lstm1 = []
        for i in range(seq_size[1]):
            reverse_seq_f_lstm1.append(seq_f_lstm1[:, seq_size[1]-i-1, :, :, :])
        reverse_seq_f_lstm1 = torch.stack(reverse_seq_f_lstm1, dim=1)
        # f3->f2->f1
        layer_output_list, _ = self.ConvLSTM_reverse(reverse_seq_f_lstm1)
        reverse_seq_f_lstm2 = layer_output_list[0]  # num_layer=1
        # reverse seq
        seq_f_lstm2 = []
        for i in range(seq_size[1]):
            seq_f_lstm2.append(reverse_seq_f_lstm2[:, seq_size[1]-i-1, :, :, :])
        seq_f_lstm2 = torch.stack(seq_f_lstm2, dim=1)
        # sum
        seq_f_lstm_sum = seq_f_lstm1 + seq_f_lstm2
        return seq_f_lstm_sum

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