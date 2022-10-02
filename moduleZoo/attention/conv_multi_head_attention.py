from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ..convolution import ConvNormActivation1d, ConvNormActivation2d
from .utils import split_cat


class MultiHeadAttention2d(nn.Module):
    """ Multi HeadSelf attention Layer"""
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 y_in_channels: Optional[int] = None,
                 y_out_channels: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: Union[int, Tuple[int, int]] = 1,
                 interpolation_mode: Optional[str] = 'nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = self.in_channels if y_in_channels is None else y_in_channels
        self.y_out_channels = self.out_channels if y_out_channels is None else y_out_channels

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode

        self.out_channels *= n_heads
        self.y_out_channels *= n_heads

        self.query_conv = ConvNormActivation2d(self.in_channels, self.out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support
        self.key_conv = ConvNormActivation2d(self.in_channels, self.out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.value_conv = ConvNormActivation2d(self.y_in_channels, self.y_out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        # gamma as the shape of expanded dims with n_heads, so [n_heads, 1, _, _, ...]
        self.gamma = nn.Parameter(torch.rand((n_heads, 1, 1) if n_heads == 1 else (n_heads, 1, 1, 1))+0.001) if self.residual else None

        self.proj = 'id' if self.y_out_channels//self.n_heads == self.y_in_channels and not self.residual else 'projection'
        self.projection = ConvNormActivation2d(self.y_in_channels, self.y_out_channels//n_heads, 1, bias=False) if self.proj == 'projection' else None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X C X W X H)
                y : feature map attention to be applied
            returns :
                out : self attention value or + input feature
        """
        b, c, w, h = x.size()
        B, C, W, H = y.size()

        assert(b == B)
        if w != W and h != H:
            assert(self.interpolation_mode is not None)
            x = interpolate(x, (W, H), mode=self.interpolation_mode) # B, c, W, H

        proj_query = self.query_conv(x).view(B, -1, W*H) # B X (C/r*n) X N
        proj_key = self.key_conv(x).view(B, -1, W*H) # B X (C/r*n) X N

        proj_value = self.value_conv(y).view(B, -1, W*H) # B X (OC*n) X N

        if self.n_heads != 1:
            split_size = [self.out_channels//self.n_heads, self.out_channels//self.n_heads, self.y_out_channels//self.n_heads]
            # n_heads*B X (C/r) X N, n_heads*B X (C/r) X N, n_heads*B X OC X N
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 1, 0), split_cat(proj_key, split_size[1], 1, 0), split_cat(proj_value, split_size[2], 1, 0)

        proj_query = proj_query.permute(0, 2, 1) # n_heads*B X N X (C/r)

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B X N X N
        attention = self.softmax(energy) # n_heads*B X N X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # n_heads*B X OC X N

        if self.residual:
            if self.projection is not None:
                y = self.projection(y).view(B, -1, W*H) # B X OC X W*H

            if self.n_heads != 1:
                out = split_cat(out, B, 0, -1) # n_heads X B X OC X N
                y = y.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1) # n_head X B X OC X N

            out = (self.gamma*out + y)/(1 + self.gamma)

            if self.n_heads != 1:
                out = split_cat(out, 1, 0, 2).squeeze(dim=0) # B X OC*n_heads X N

            out = out.view(B, self.out_channels, W, H) # B X OC*n_heads X W X H

            return out

        out = split_cat(out, B, 0, 1) # B X OC*n_heads X N
        out = out.view(B, self.y_out_channels, W, H) # B X OC*n_heads X W X H

        return out

    def shape(self, in_shape: Tuple[int, int], y_in_shape: Tuple[int, int]):
        return y_in_shape


class MultiHeadAttention1d(nn.Module):
    """ Multi HeadSelf attention Layer"""
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 y_in_channels: Optional[int] = None,
                 y_out_channels: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: Union[int, Tuple[int, int]] = 1,
                 interpolation_mode: Optional[str] = 'nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = self.in_channels if y_in_channels is None else y_in_channels
        self.y_out_channels = self.out_channels if y_out_channels is None else y_out_channels

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode

        self.out_channels *= n_heads
        self.y_out_channels *= n_heads

        self.query_conv = ConvNormActivation1d(self.in_channels, self.out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support
        self.key_conv = ConvNormActivation1d(self.in_channels, self.out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.value_conv = ConvNormActivation1d(self.y_in_channels, self.y_out_channels, kernel_size, padding='stride_effective') # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        # gamma as the shape of expanded dims with n_heads, so [n_heads, 1, _, _, ...]
        self.gamma = nn.Parameter(torch.rand((n_heads, 1, 1) if n_heads == 1 else (n_heads, 1, 1, 1))+0.001) if self.residual else None

        self.proj = 'id' if self.y_out_channels//self.n_heads == self.y_in_channels and not self.residual else 'projection'
        self.projection = ConvNormActivation1d(self.y_in_channels, self.y_out_channels//self.n_heads, 1, bias=False) if self.proj == 'projection' else None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X C X N)
                y : feature map attention to be applied
            returns :
                out : self attention value or + input feature
        """
        b, c, n = x.size()
        B, C, N = y.size()

        assert(b == B)
        if n != N:
            assert(self.interpolation_mode is not None)
            x = interpolate(x, N, mode=self.interpolation_mode) # B, c, N

        proj_query = self.query_conv(x) # B X (C/r*n) X N
        proj_key = self.key_conv(x) # B X (C/r*n) X N

        proj_value = self.value_conv(y) # B X (OC*n) X N

        if self.n_heads != 1:
            split_size = [self.out_channels//self.n_heads, self.out_channels//self.n_heads, self.y_out_channels//self.n_heads]
            # n_heads*B X (C/r) X N, n_heads*B X (C/r) X N, n_heads*B X OC X N
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 1, 0), split_cat(proj_key, split_size[1], 1, 0), split_cat(proj_value, split_size[2], 1, 0)

        proj_query = proj_query.permute(0, 2, 1) # n_heads*B X N X (C/r)

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B X N X N
        attention = self.softmax(energy) # n_heads*B X N X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # n_heads*B X OC X N

        if self.residual:
            if self.projection is not None:
                y = self.projection(y) # B X OC X N

            if self.n_heads != 1:
                out = split_cat(out, B, 0, -1) # n_heads X B X OC X N
                y = y.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1) # n_head X B X OC X N

            out = (self.gamma*out + y)/(1 + self.gamma)

            if self.n_heads != 1:
                out = split_cat(out, 1, 0, 2).squeeze(dim=0) # B X OC*n_heads X N

            return out

        out = split_cat(out, B, 0, 1) # B X OC*n_heads X N

        return out

    def shape(self, in_shape: int, y_in_shape: int):
        return y_in_shape


class MultiHeadSelfAttention2d(MultiHeadAttention2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: Union[int, Tuple[int, int]] = 1):
        super().__init__(in_channels, out_channels, None, None, n_heads, residual, kernel_size, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, x)

class MultiHeadSelfAttention1d(MultiHeadAttention1d):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: Union[int, Tuple[int, int]] = 1):
        super().__init__(in_channels, out_channels, None, None, n_heads, residual, kernel_size, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, x)
