from Models.Wrappers import Wrapper

import torch


class MLPWrapper(Wrapper):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        model_name: str,
        model_type: str,
        parameters_path: str = './Models/Parameters/',
        layers: int = 3,
        **kwargs,
        ):

        Wrapper.__init__(
            self, 
            in_channels,
            out_channels,
            model_name,
            model_type,
            parameters_path)

        self.layer_count = layers
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU()))

        for _ in range(1, self.layer_count - 1):
            self.layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU()))

        self.layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_channels, out_channels)))

        self.load_parameters()

    def forward(self, x: torch.Tensor):

        out = self.layers[0](x)

        for i in range(1, self.layer_count - 1):
            out = self.layers[i](out)

        out = self.layers[-1](out)

        return out

