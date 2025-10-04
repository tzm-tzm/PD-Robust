import torch
import torch.nn.init as init
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
from utils import cal_centr

class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        res = x
        x = self.activation(self.fc(x))
        x = x + res
        return x


class MyDenseLayer(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(MyDenseLayer, self).__init__()
        self.fc1 = ResBlock(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.num_layers = num_layers
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

class MyCLModel(nn.Module):
    #bin_idx为[5000, 1]的矩阵，储存每个向量在哪个段
    def __init__(self, in_dim, out_dim,
                 loss_function=None,
                 num_layers=1,
                 device="cuda:0"
                 ):
        super(MyCLModel, self).__init__()
        self.layer = MyDenseLayer(in_dim, out_dim, num_layers=num_layers)
        self.loss_function = loss_function
        self.device=device

    def forward(self, x, bin_idx, alpha, bin_num):
        h = self.layer(x)

        sample_embed = h

        bin_centr, sample_centr = cal_centr(bin_num, sample_embed, bin_idx, self.device)

        loss = self.loss_function(sample_embed, bin_centr, sample_centr, alpha)

        return h, loss, bin_centr







