from torch import nn, Tensor
from timm.models.layers import to_2tuple
from typing import Tuple, Optional, Union, List


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self, 
        img_size: Union[tuple, int] = 224, 
        patch_size: Union[tuple, int] = 16, 
        in_chans = 1, 
        embed_dim = 768, 
        norm_layer = None, 
        flatten = True):
        super().__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)

        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        # 输入通道，输出通道，卷积核大小，步长
        # C*H*W->embed_dim*grid_size*grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
 
    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x