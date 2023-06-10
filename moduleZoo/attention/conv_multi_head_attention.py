from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ..convolution import ConvNormActivation1d, ConvNormActivation2d
from .utils import split_cat


class MultiHeadAttention2d(nn.Module):
    """ Multi HeadSelf attention Layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int | None = None,
                 y_in_channels: int | None = None,
                 y_out_channels: int | None = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: int | Tuple[int, int] = 1,
                 interpolation_mode: str | None = 'nearest',
                 channel_cross_attention: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = self.in_channels if y_in_channels is None else y_in_channels
        self.y_out_channels = self.out_channels if y_out_channels is None else y_out_channels

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode
        self.chxa = channel_cross_attention

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

    def extract_qkv(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.query_conv(x), self.key_conv(x), self.value_conv(y) # [B*k X (C | OC*n) X H X W]
        return q, k, v

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                proj_query: torch.Tensor | None = None,
                proj_key: torch.Tensor | None = None,
                proj_value: torch.Tensor | None = None) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X k X C X W X H)
                y : feature map attention to be applied ( B X k X C X W_o X H_o)
            returns :
                out : self attention value or + input feature
        """
        final_squeeze = False

        if x.ndim == 4:
            x = x.unsqueeze(dim=1)
            final_squeeze = True

        if y.ndim == 4:
            y = y.unsqueeze(dim=1)
            final_squeeze = True

        assert(x.ndim == 5 and y.ndim == 5)

        b, k, c, h, w = x.size()
        B, K, C, H, W = y.size()

        assert(b == B and k == K)

        x, y = x.view(b*k, c, h, w), y.view(B*K, c, H, W)

        if h != H or w != W:
            assert(self.interpolation_mode is not None)
            x = interpolate(x, (H, W), mode=self.interpolation_mode) # [B*k, c, H, W]

        if (proj_query is None or proj_key is None or proj_value is None):
            proj_query, proj_key, proj_value = self.extract_qkv(x, y)

            # [B X k X (C/r)*n_heads X H X W], [B X k X (C/r)*n_heads X H X W], [B X k X (OC)*n_heads X H X W]
            proj_query, proj_key, proj_value = \
                proj_query.view(B, k, self.out_channels, H, W), \
                    proj_key.view(B, k, self.out_channels, H, W), \
                        proj_value.view(B, K, self.y_out_channels, H, W)

        # [B X (C/r)*n_heads X k X N], [B X (C/r)*n_heads X k X N], [B X (OC)*n_heads X k X N]
        proj_query, proj_key, proj_value = \
            proj_query.flatten(start_dim=-2).permute(0, 2, 1, 3), \
                proj_key.flatten(start_dim=-2).permute(0, 2, 1, 3), \
                    proj_value.flatten(start_dim=-2).permute(0, 2, 1, 3)

        if self.n_heads != 1:
            split_size = [self.out_channels//self.n_heads, self.out_channels//self.n_heads, self.y_out_channels//self.n_heads]
            # n_heads*B X (C/r) X k X N, n_heads*B X (C/r) X k X N, n_heads*B X OC X k X N
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 1, 0), split_cat(proj_key, split_size[1], 1, 0), split_cat(proj_value, split_size[2], 1, 0)

        if self.chxa:
            st_dim, end_dim = 1, 2
        else:
            st_dim, end_dim = 0, 1

        proj_query, proj_key, proj_value = \
            proj_query.flatten(st_dim, end_dim), \
                proj_key.flatten(st_dim, end_dim), \
                    proj_value.flatten(st_dim, end_dim)

        proj_query = proj_query.permute(0, 2, 1) # n_heads*B*C'' X N X C'

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B*C'' X N X N
        attention = self.softmax(energy) # n_heads*B X N X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # n_heads*B X OC X N

        # Final output reshape
        out = out.view(self.n_heads*B, self.y_out_channels//self.n_heads, K, H*W) # [n_heads*B X OC X K X N]

        out = split_cat(out, B, 0, -1) # [n_heads X B X OC X K X N]

        if self.residual:
            if self.projection is not None:
                y = self.projection(y).view(B, K, -1, H*W).permute(0, 2, 1, 3) # B X OC X K X W*H
                y = y.flatten(st_dim, end_dim) # [B*OC X K X W*H] | # [B X OC*K X W*H]

            y = y.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1, 1) # [n_head X B X OC X K X N]

            out = (self.gamma*out + y)/(1 + self.gamma)

        out = out.permute(1, 0, 3, 2, 4).view(B, self.n_heads, self.y_out_channels, H, W) # [B X n_heads X K X OC X H X W]

        if final_squeeze:
            out = out.squeeze(dim=2) # [B X n_heads X OC X H X W]
            if self.n_heads == 1:
                out = out.squeeze(dim=1) # [B X OC X H X W]

        return out

    def shape(self, in_shape: Tuple[int, int], y_in_shape: Tuple[int, int]):
        return y_in_shape


class MultiHeadAttention1d(nn.Module):
    """ Multi HeadSelf attention Layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 y_in_channels: Optional[int] = None,
                 y_out_channels: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 kernel_size: Union[int, Tuple[int, int]] = 1,
                 interpolation_mode: Optional[str] = 'nearest',
                 channel_cross_attention: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = self.in_channels if y_in_channels is None else y_in_channels
        self.y_out_channels = self.out_channels if y_out_channels is None else y_out_channels

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode
        self.chxa = channel_cross_attention

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

    def extract_qkv(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.query_conv(x), self.key_conv(x), self.value_conv(y) # [B*k X (C*n_heads | OC*n_heads) X N]
        return q, k, v

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                proj_query: torch.Tensor | None = None,
                proj_key: torch.Tensor | None = None,
                proj_value: torch.Tensor | None = None) -> torch.Tensor:
        """
            inputs :
                x : input feature maps( B X K X C X N)
                y : feature map attention to be applied
            returns :
                out : self attention value or + input feature
        """
        final_squeeze = False

        if x.ndim == 3:
            x = x.unsqueeze(dim=1)
            final_squeeze = True

        if y.ndim == 3:
            y = y.unsqueeze(dim=1)
            final_squeeze = True

        assert(x.ndim == 4 and y.ndim == 4)

        b, k, c, n = x.size()
        B, K, C, N = y.size()

        assert(b == B and k == K)

        x, y = x.view(b*k, c, n), y.view(B*K, c, N)

        if n != N:
            assert(self.interpolation_mode is not None)
            x = interpolate(x, N, mode=self.interpolation_mode) # [B, k, c, N]

        if (proj_query is None or proj_key is None or proj_value is None):
            proj_query, proj_key, proj_value = self.extract_qkv(x, y)

            # [B X k X (C/r)*n_heads X N], [B X k X (C/r)*n_heads X N], [B X k X (OC)*n_heads X N]
            proj_query, proj_key, proj_value = \
                proj_query.view(B, k, self.out_channels, N), \
                    proj_key.view(B, k, self.out_channels, N), \
                        proj_value.view(B, K, self.y_out_channels, N)

        # [B X (C/r)*n_heads X k X N], [B X (C/r)*n_heads X k X N], [B X (OC)*n_heads X K X N]
        proj_query, proj_key, proj_value = \
            proj_query.permute(0, 2, 1, 3), \
                proj_key.permute(0, 2, 1, 3), \
                    proj_value.permute(0, 2, 1, 3)

        if self.n_heads != 1:
            split_size = [self.out_channels//self.n_heads, self.out_channels//self.n_heads, self.y_out_channels//self.n_heads]
            # n_heads*B X (C/r) X k X N, n_heads*B X (C/r) X k X N, n_heads*B X OC X K X N
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 1, 0), split_cat(proj_key, split_size[1], 1, 0), split_cat(proj_value, split_size[2], 1, 0)

        if self.chxa:
            st_dim, end_dim = 1, 2
        else:
            st_dim, end_dim = 0, 1

        proj_query, proj_key, proj_value = \
            proj_query.flatten(st_dim, end_dim), \
                proj_key.flatten(st_dim, end_dim), \
                    proj_value.flatten(st_dim, end_dim)

        proj_query = proj_query.permute(0, 2, 1) # n_heads*B X N X (C/r)

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B X N X N
        attention = self.softmax(energy) # n_heads*B X N X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # n_heads*B X OC X N

        # Final output reshape
        out = out.view(self.n_heads*B, self.y_out_channels//self.n_heads, K, N) # [n_heads*B X OC X K X N]

        out = split_cat(out, B, 0, -1) # [n_heads X B X OC X K X N]

        if self.residual:
            if self.projection is not None:
                y = self.projection(y).view(B, K, -1, N).permute(0, 2, 1, 3) # B X OC X K X W*H
                y = y.flatten(st_dim, end_dim) # [B*OC X K X W*H] | # [B X OC*K X W*H]

            y = y.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1, 1) # [n_head X B X OC X K X N]

            out = (self.gamma*out + y)/(1 + self.gamma)

            return out

        out = out.permute(1, 0, 3, 2, 4) # [B X n_heads X K X OC X N]

        if final_squeeze:
            out = out.squeeze(dim=2) # [B X n_heads X OC X N]
            if self.n_heads == 1:
                out = out.squeeze(dim=1) # [B X OC X N]

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
