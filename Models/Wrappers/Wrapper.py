import os
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from Models.utils import time_handlers

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

    @time_handlers.timer
    def load_parameters(self):
        
        if os.path.exists(f'{self.path}/{self.model_name}_{self.model_type}.pth'):
            self.load_state_dict(torch.load(f'{self.path}/{self.model_name}_{self.model_type}.pth'))
        
    @time_handlers.timer
    def save_parameters(self):
        
        torch.save(self.state_dict(), f'{self.path}/{self.model_name}_{self.model_type}.pth')

    def forward(self, x):
        pass

    def reset_parameters(self):
        pass

    def loss(self, pred, label):
        return F.mse_loss(pred, label)