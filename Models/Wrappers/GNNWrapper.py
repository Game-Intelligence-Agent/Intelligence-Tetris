from turtle import forward

from torch import dtype

from Models.Wrappers import Wrapper
from Models.Modules import PatchEmbed

import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.glob import  global_add_pool, global_max_pool, global_mean_pool, Set2Set, GlobalAttention

from typing import Tuple, Optional, Union, List


class GATv2Wrapper(Wrapper):

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
        add_self_loop: bool = True,
        positional_embedding: bool = True,
        img_size: Union[tuple, int, list] = (20, 10), 
        patch_size: Union[tuple, int, list] = (2 ,2),
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

        if self.model is None:

            self.model = torch.nn.ModuleDict()
            self.model['patch_embed'] = PatchEmbed(img_size = img_size, patch_size = patch_size, embed_dim = embed_dim, in_chans = in_channels)
            self.num_patches = self.model['patch_embed'].num_patches
            self.embed_dim = embed_dim
            self.model['Parameters'] = torch.nn.ParameterDict()
            self.model['Parameters']['pos_embed'] = torch.nn.Parameter(torch.zeros(in_channels, self.num_patches, embed_dim)) if positional_embedding else torch.nn.Parameter(torch.zeros(in_channels, num_patches, embed_dim), requires_grad = False)

            for idx in range(1, layers + 1):

                self.model[f'GraphConv{idx}'] = GATv2Conv(in_channels = embed_dim, out_channels = embed_dim, heads = heads, concat = False, dropout = dropout, negative_slope = negative_slope, bias = bias, add_self_loops = add_self_loop)
                self.model[f'activation{idx}'] = torch.nn.Sequential(torch.nn.LeakyReLU(negative_slope = negative_slope))

            self.model['after_mapping'] = torch.nn.Sequential(torch.nn.Linear(embed_dim, out_channels, bias = bias))
            if pooling == 'set2set':
                self.model['readout'] = Set2Set(embed_dim, processing_steps = 2)
                self.model['after_mapping'] = torch.nn.Sequential(torch.nn.Linear(2 * embed_dim, out_channels))
            elif pooling == 'attention':
                self.model['readout'] = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim, bias = bias), torch.nn.BatchNorm1d(embed_dim), torch.nn.ReLU(), torch.nn.Linear(embed_dim, 1, bias = bias)))
            elif pooling == 'add':
                self.model['readout'] = global_add_pool
            elif pooling == 'mean':
                self.model['readout'] = global_mean_pool
            elif pooling == 'max':
                self.model['readout'] = global_max_pool
                
        else:

            self.num_patches = self.model['patch_embed'].num_patches
            self.embed_dim = self.model['patch_embed'].embed_dim
        
        self.adj = torch.ones((self.num_patches, self.num_patches))

    def forward(self, x: torch.Tensor):

        # print(x.shape)

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        # print(x.shape)

        size = x.shape[0]
            
        x = self.model['patch_embed'](x)
        x = x + self.model['Parameters']['pos_embed']

        # print(x.shape)

        x = x.reshape(-1, self.embed_dim)
        edge_index = torch.diag(torch.ones(size)).kron(self.adj).nonzero().t().contiguous().to(x.device)
        # print(x.shape)
        # print(edge_index.shape)

        out = x
        for idx in range(1, self.layers + 1):

            residule = self.model[f'GraphConv{idx}'](out, edge_index)
            residule = self.model[f'activation{idx}'](residule)
            out = residule + out
        out = out + x

        # print(out.shape)

        batch = torch.LongTensor([[i] * self.num_patches for i in range(0, size)]).view(-1,).to(x.device)

        # print(batch.shape)

        out = self.model['readout'](out, batch)
        # print(out.shape)
        out = self.model['after_mapping'](out)
        # print(out.shape)

        return out

