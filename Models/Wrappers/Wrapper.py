import os
from collections import namedtuple

import torch
from torch import nn
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple

from utils import time_handlers

class Wrapper(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_name: str,
        model_type: str,
        parameters_path: str = './Models/Parameters/',
        ):
        super(Wrapper, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_name = model_name
        self.model_type = model_type
        self.path = parameters_path
        self.model = None

    def build_model(self):
        pass

    @time_handlers.timer
    def load_parameters(self):
        
        self.model = torch.load(f'{self.path}/{self.model_name}_{self.model_type}.pth')
        
    @time_handlers.timer
    def save_parameters(self, prefix = ''):
        
        torch.save(self.model, f'{self.path}/{self.model_name}_{self.model_type}.pth')

    def forward(self, x):
        pass

    def loss(self):
        pass