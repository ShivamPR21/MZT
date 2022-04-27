from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .utils import split_cat


class MultiHeadAttentionLinear(nn.Module):
    """ Multi HeadSelf attention Layer"""
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 y_in_dim: Optional[int] = None,
                 y_out_dim: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 interpolation_mode: Optional[str] = 'nearest'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        self.y_in_dim = self.in_dim if y_in_dim is None else y_in_dim
        self.y_out_dim = self.out_dim if y_out_dim is None else y_out_dim

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode

        self.out_dim *= n_heads
        self.y_out_dim *= n_heads

        self.query_conv = nn.Linear(self.in_dim, self.out_dim)
        self.key_conv = nn.Linear(self.in_dim, self.out_dim)

        self.value_conv = nn.Linear(self.y_in_dim, self.y_out_dim)

        self.gamma = nn.Parameter(torch.rand(n_heads, 1, 1, 1)+0.001) if self.residual else None

        self.proj = 'id' if self.y_out_dim//self.n_heads == self.y_in_dim and not self.residual else 'projection'
        self.projection = nn.Linear(self.y_in_dim, self.y_out_dim//self.n_heads, bias=False) if self.proj == 'projection' else None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X C X N)
                y : feature map attention to be applied
            returns :
                out : self attention value or + input feature
        """
        b, n = x.size()
        B, N = y.size()

        assert(b == B)

        proj_query = self.query_conv(x).unsqueeze(dim=1) # B X 1 X N * n_heads
        proj_key = self.key_conv(x).unsqueeze(dim=1) # B X 1 X N * n_heads

        proj_value = self.value_conv(y).unsqueeze(dim=1) # B X 1 X ON * n_heads

        if self.n_heads != 1:
            split_size = [self.out_dim//self.n_heads, self.out_dim//self.n_heads, self.y_out_dim//self.n_heads]
            # n_heads*B X 1 X N, n_heads*B X 1 X N, n_heads*B X 1 X ON
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 2, 0), \
                    split_cat(proj_key, split_size[1], 2, 0), \
                        split_cat(proj_value, split_size[2], 2, 0)

        proj_query = proj_query.transpose(2, 1) # n_heads*B X N X 1

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B X N X N
        attention = self.softmax(energy) # n_heads*B X N X N

        if self.out_dim != self.y_out_dim:
            assert(self.interpolation_mode is not None)
            attention = interpolate(attention,
                                    (self.y_out_dim, self.y_out_dim),
                                    mode=self.interpolation_mode) # B, ON, ON

        out = torch.bmm(proj_value, attention.transpose(2, 1)) # n_heads*B X 1 X ON

        if self.residual:
            if self.projection is not None:
                y = self.projection(y).unsqueeze(dim=1) # B X 1 X ON

            if self.n_heads != 1:
                out = split_cat(out, B, 0, -1) # n_heads X B X 1 X ON
                y = y.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1) # n_head X B X 1 X ON

            out = (self.gamma*out + y)/(1 + self.gamma)

            if self.n_heads != 1:
                out = split_cat(out, 1, 0, 2).squeeze(dim=0) # B X n_heads X ON

            return out

        out = split_cat(out, B, 0, 1) # B X n_heads X ON

        return out

    def shape(self, in_shape: int, y_in_shape: int):
        return y_in_shape
