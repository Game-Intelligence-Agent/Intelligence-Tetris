from turtle import forward

from Models.Wrappers import Wrapper
from Models.Modules import PatchEmbed

import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.glob import  global_add_pool, global_max_pool, global_mean_pool, Set2Set, GlobalAttention

from typing import Tuple, Optional, Union, List


class GATWrapper(Wrapper):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        model_name: str,
        model_type: str,
        dropout: float = 0.2,
        negative_slope: float = 0.2, 
        parameters_path: str = './Models/Parameters/',
        bias: bool = True,
        # add_self_loop: bool = True,
        positional_embedding: bool = True,
        img_size: Union[tuple, int] = (20, 10), 
        patch_size: Union[tuple, int] = (2 ,2),
        embed_dim: int = 32,
        layers: int = 3,
        pooling: str = 'none',
        **kwargs,
        ):

        Wrapper.__init__(
            self, 
            in_channels,
            out_channels,
            model_name,
            model_type,
            parameters_path)

        self.load_parameters()

        self.layers = layers
        self.pooling = pooling

        if self.model is None:

            self.model = torch.nn.ModuleDict()
            self.model['patch_embed'] = PatchEmbed(img_size = img_size, patch_size = patch_size, embed_dim = embed_dim)
            num_patches = self.model['patch_embed'].num_patches
            self.model['pos_embed'] = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if positional_embedding else torch.zeros(1, num_patches, embed_dim)

            for idx in range(1, layers + 1):

                self.model[f'GraphConv{idx}'] = GATv2Conv(in_channels = embed_dim, out_channels = embed_dim, heads = heads, concat = False, dropout = dropout, negative_slope = negative_slope)
                self.model[f'activation{idx}'] = torch.nn.Sequential(torch.nn.LeakyReLU(negative_slope = negative_slope))

            self.model['after_mapping'] = torch.nn.Sequential(torch.nn.Linear(embed_dim, out_channels))
            if pooling == 'set2set':
                self.model['readout'] = Set2Set(embed_dim, processing_steps = 2)
                self.model['after_mapping'] = torch.nn.Sequential(torch.nn.Linear(2 * embed_dim, out_channels))
            elif pooling == 'attention':
                self.model['readout'] = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.BatchNorm1d(embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 1)))
            if self.pooling == 'add':
                self.model['readout'] = global_add_pool
            elif self.pooling == 'mean':
                self.model['readout'] = global_mean_pool
            elif self.pooling == 'max':
                self.model['readout'] = global_max_pool
                
        else:

            num_patches = self.model['patch_embed'].num_patches
        
        self.edge_index = torch.ones((num_patches, num_patches)).nonzero().t().contiguous()

    def forward(self, x: torch.Tensor, batch):

        if len(x.shape) == 3:
            x.squeeze(1)

        x = self.model['patch_embed'](x)
        x = x + self.model['pos_embed']
        
        out = x
        for idx in range(1, self.layers + 1):

            residule = self.model[f'GraphConv{idx}'](out, self.edge_index)
            residule = self.model[f'activation{idx}'](residule)
            out = residule + out
        out = out + x

        out = self.model['readout'](out)
        out = self.model['after_mapping'](out)

        return out

