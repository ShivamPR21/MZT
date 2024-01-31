from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ..convolution import ConvNormActivation1d, ConvNormActivation2d
from ..utils import NestedTensor
from .utils import split_cat


class MultiHeadAttention2d(nn.Module):
    """Multi HeadSelf attention Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        y_in_channels: int | None = None,
        y_out_channels: int | None = None,
        n_heads: int = 1,
        residual: bool = True,
        kernel_size: int | Tuple[int, int] = 1,
        interpolation_mode: str | None = "nearest",
        attention_type: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = (
            self.in_channels if y_in_channels is None else y_in_channels
        )
        self.y_out_channels = (
            self.out_channels if y_out_channels is None else y_out_channels
        )

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode
        if attention_type not in (0, 1, 2):
            raise ValueError(
                "Parameter `attention_type` should be one of the (0, 1, 2)"
            )
        self.xa = attention_type  # TODO@ShivamPR21: Supply Better Name

        self.out_channels *= n_heads
        self.y_out_channels *= n_heads

        self.query_conv = ConvNormActivation2d(
            self.in_channels, self.out_channels, kernel_size, padding="stride_effective"
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support
        self.key_conv = ConvNormActivation2d(
            self.in_channels, self.out_channels, kernel_size, padding="stride_effective"
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.value_conv = ConvNormActivation2d(
            self.y_in_channels,
            self.y_out_channels,
            kernel_size,
            padding="stride_effective",
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        # gamma as the shape of expanded dims with n_heads, so [n_heads, 1, _, _, ...]
        self.gamma = (
            nn.Parameter(
                torch.rand((n_heads, 1, 1) if n_heads == 1 else (n_heads, 1, 1, 1))
                + 0.001
            )
            if self.residual
            else None
        )

        self.proj = (
            "id"
            if self.y_out_channels // self.n_heads == self.y_in_channels
            and not self.residual
            else "projection"
        )
        self.projection = (
            ConvNormActivation2d(
                self.y_in_channels, self.y_out_channels // n_heads, 1, bias=False
            )
            if self.proj == "projection"
            else None
        )

        self.softmax = nn.Softmax(dim=-1)

    def extract_qkv(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (
            self.query_conv(x),
            self.key_conv(y),
            self.value_conv(y),
        )  # [B*k X (C | OC*n) X H X W]
        return q, k, v

    def forward(
        self,
        x: NestedTensor | torch.Tensor,
        y: NestedTensor | torch.Tensor,
        proj_query: NestedTensor | torch.Tensor | None = None,
        proj_key: NestedTensor | torch.Tensor | None = None,
        proj_value: NestedTensor | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs :
            x : input feature maps( B X k_q X C X W X H)
            y : feature map attention to be applied ( B X k_v X C X W_o X H_o)
        returns :
            out : self attention value or + input feature
        """
        mask = x.mask
        x, y = x.data, y.data

        final_squeeze = False

        if x.ndim == 4:
            x = x.unsqueeze(dim=1)
            final_squeeze = True

        if y.ndim == 4:
            y = y.unsqueeze(dim=1)
            final_squeeze = True

        if not (x.ndim == 5 and y.ndim == 5):
            raise ValueError(f"Got {x.ndim = } and {y.ndim = } should be 5")

        b, k, c, h, w = x.size()
        B, K, C, H, W = y.size()
        N = H * W

        if not (b == B):
            raise ValueError(
                f"Batch size of both input doesn't match, given {b = } and {B = }"
            )

        x, y = x.view(b * k, c, h, w), y.view(B * K, c, H, W)

        if h != H or w != W:
            if self.interpolation_mode is None:
                raise ValueError("Interpolation mode expected.")

            x = interpolate(x, (H, W), mode=self.interpolation_mode)  # [B*k, c, H, W]

        if proj_query is None or proj_key is None or proj_value is None:
            proj_query, proj_key, proj_value = self.extract_qkv(x, y)

            # [B X k_q X (C/r)*n_heads X H X W], [B X k_k X (C/r)*n_heads X H X W], [B X k_v == k_k X (OC)*n_heads X H X W]
            proj_query, proj_key, proj_value = (
                proj_query.view(B, k, self.out_channels, H, W),
                proj_key.view(B, K, self.out_channels, H, W),
                proj_value.view(B, K, self.y_out_channels, H, W),
            )

        # [B X k_q X N X (C/r)*n_heads], [B X k_k X N X (C/r)*n_heads], [B X k_v == k_k X N X (OC)*n_heads]
        proj_query, proj_key, proj_value = (
            proj_query.flatten(start_dim=-2).transpose(2, 3),
            proj_key.flatten(start_dim=-2).transpose(2, 3),
            proj_value.flatten(start_dim=-2).transpose(2, 3),
        )

        if self.n_heads != 1:
            split_size = [
                self.out_channels // self.n_heads,
                self.out_channels // self.n_heads,
                self.y_out_channels // self.n_heads,
            ]

            # n_heads X B X k_q X N X (C/r), n_heads X B X k_k X N X (C/r), n_heads X B X k_v == k_k X N X OC
            proj_query, proj_key, proj_value = (
                split_cat(proj_query, split_size[0], -1, -1),
                split_cat(proj_key, split_size[1], -1, -1),
                split_cat(proj_value, split_size[2], -1, -1),
            )

        # TODO@ShivamPR21: Added new level of attention, check for integrity
        if self.xa == 0:
            st_dim, end_dim = 1, 2
            if k != K:
                raise ValueError(
                    "For no cross attention, the elements in x and y should be same s.t. k == K"
                )
            # n_heads X B*k_q X N X (C/r), n_heads X B*k_k X N X (C/r), n_heads X B*(k_v == k_k) X N X OC
            proj_query, proj_key, proj_value = (
                proj_query.flatten(st_dim, end_dim),
                proj_key.flatten(st_dim, end_dim),
                proj_value.flatten(st_dim, end_dim),
            )
            if mask is not None and mask.shape[-2:] != [N, N]:
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )
        elif self.xa == 1:
            st_dim, end_dim = 2, 3
            # n_heads X B X (k_q*N) X (C/r), n_heads X B X (k_k*N) X (C/r), n_heads X B X ((k_v == k_k)*N) X OC
            proj_query, proj_key, proj_value = (
                proj_query.flatten(st_dim, end_dim),
                proj_key.flatten(st_dim, end_dim),
                proj_value.flatten(st_dim, end_dim),
            )
            if mask is not None and mask.shape[-2:] != [k * N, K * N]:
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )
        else:
            st_dim, end_dim = 1, 2
            # n_heads X B*N X (k_q) X (C/r), n_heads X B*N X (k_k) X (C/r), n_heads X B*N X (k_v == k_k) X OC
            proj_query, proj_key, proj_value = (
                proj_query.transpose(3, 2).flatten(st_dim, end_dim),
                proj_key.transpose(3, 2).flatten(st_dim, end_dim),
                proj_value.transpose(3, 2).flatten(st_dim, end_dim),
            )
            if mask is not None and mask.shape[-2:] != [k, K]:
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )

        # [n_heads, B, C', (k_k * N)] or [n_heads, B*k_k, C', (N)] or [n_heads, B*N, C', (k_k)]
        proj_key = proj_key.transpose(3, 2)

        proj_query, proj_key, proj_value = (
            proj_query.flatten(0, 1),  # n_heads * B X , ...
            proj_key.flatten(0, 1),  # n_heads * B X ...
            proj_value.flatten(0, 1),  # n_heads * B X ...
        )

        # n_heads * B X (k_q * N) X (k_k * N) or n_heads * B*(k_q == k_k) X (N) X (N) or n_heads * B*N X (k_q) X (k_k)
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        # Mask out unwanted attentions
        if mask is not None:
            # mask is of shape =>
            # (k_q * N) X (k_k * N) or (N) X (N) or (k_q) X (k_k)
            energy[..., ~mask] = -torch.inf  # TODO@ShivamPR21: Check for integrity

        # n_heads * B X (k_q * N) X (k_k * N) or n_heads * B*(k_q == k_k) X (N) X (N) or n_heads * B*N X (k_q) X (k_k)
        attention = self.softmax(energy)

        # [n_heads * B, (k_q * N), OC] or [n_heads * B*k_k, N, OC] or [n_heads * B*N, (k_q), OC]
        out = torch.bmm(attention, proj_value)

        # Final output reshape
        # [n_heads X B X k X N X OC]
        if self.xa == 2:
            out = out.view(
                self.n_heads, -1, N, k, self.y_out_channels // self.n_heads
            ).transpose(
                3, 2
            )  # [n_heads X B X N X k_q X OC] => [n_heads X B X k_q X N X OC]
        else:
            out = out.view(
                self.n_heads, -1, k, N, self.y_out_channels // self.n_heads
            )  # [n_heads X B X k_q X N X OC]

        # out = split_cat(out, B, 0, -1)  # [n_heads X B X k_q X N X OC]

        if self.residual:
            if self.projection is not None:
                x = (
                    self.projection(x).view(B, k, -1, N).transpose(2, 1)
                )  # B X k_q X N X OC

            x = x.unsqueeze(dim=0).repeat(
                self.n_heads, 1, 1, 1, 1
            )  # [n_head X B X k_q X N X OC]

            if self.gamma is None:
                raise ValueError(
                    "Trying to use gamma variable, which is not defined for this instance."
                )

            out = (self.gamma * out + x) / (1 + self.gamma)

        out = out.permute(1, 0, 2, 4, 3).view(
            B, self.n_heads, self.y_out_channels // self.n_heads, H, W
        )  # [B X n_heads X k_q X OC X H X W]

        if final_squeeze:
            out = out.squeeze(dim=2)  # [B X n_heads X OC X H X W]
            if self.n_heads == 1:
                out = out.squeeze(dim=1)  # [B X OC X H X W]

        return out

    def shape(self, in_shape: Tuple[int, int], y_in_shape: Tuple[int, int]):
        return y_in_shape


class MultiHeadAttention1d(nn.Module):
    """Multi HeadSelf attention Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        y_in_channels: Optional[int] = None,
        y_out_channels: Optional[int] = None,
        n_heads: int = 1,
        residual: bool = True,
        kernel_size: int = 1,
        interpolation_mode: str | None = "nearest",
        attention_type: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.y_in_channels = (
            self.in_channels if y_in_channels is None else y_in_channels
        )
        self.y_out_channels = (
            self.out_channels if y_out_channels is None else y_out_channels
        )

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode
        self.xa = attention_type

        self.out_channels *= n_heads
        self.y_out_channels *= n_heads

        self.query_conv = ConvNormActivation1d(
            self.in_channels, self.out_channels, kernel_size, padding="stride_effective"
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support
        self.key_conv = ConvNormActivation1d(
            self.in_channels, self.out_channels, kernel_size, padding="stride_effective"
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.value_conv = ConvNormActivation1d(
            self.y_in_channels,
            self.y_out_channels,
            kernel_size,
            padding="stride_effective",
        )  # TODO@ShivamPR21: Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        # gamma as the shape of expanded dims with n_heads, so [n_heads, 1, _, _, ...]
        self.gamma = (
            nn.Parameter(
                torch.rand((n_heads, 1, 1) if n_heads == 1 else (n_heads, 1, 1, 1))
                + 0.001
            )
            if self.residual
            else None
        )

        self.proj = (
            "id"
            if self.y_out_channels // self.n_heads == self.y_in_channels
            and not self.residual
            else "projection"
        )
        self.projection = (
            ConvNormActivation1d(
                self.y_in_channels, self.y_out_channels // self.n_heads, 1, bias=False
            )
            if self.proj == "projection"
            else None
        )

        self.softmax = nn.Softmax(dim=-1)

    def extract_qkv(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (
            self.query_conv(x),
            self.key_conv(y),
            self.value_conv(y),
        )  # [B*k X (C*n_heads | OC*n_heads) X N]
        return q, k, v

    def forward(
        self,
        x: NestedTensor | torch.Tensor,
        y: NestedTensor | torch.Tensor,
        proj_query: NestedTensor | torch.Tensor | None = None,
        proj_key: NestedTensor | torch.Tensor | None = None,
        proj_value: NestedTensor | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs :
            x : input feature maps( B X k_q X C X N)
            y : feature map attention to be applied( B X k_v X C X N)
        returns :
            out : self attention value or + input feature
        """
        mask = x.mask
        x, y = x.data, y.data

        final_squeeze = False

        if x.ndim == 3:
            x = x.unsqueeze(dim=1)
            final_squeeze = True

        if y.ndim == 3:
            y = y.unsqueeze(dim=1)
            final_squeeze = True

        if not (x.ndim == 4 and y.ndim == 4):
            raise ValueError(f"Got {x.ndim = } and {y.ndim = } should be 4")

        b, k, c, n = x.size()
        B, K, C, N = y.size()

        if not (b == B):
            raise ValueError(
                f"Batch size of both input doesn't match, given {b = } and {B = }"
            )

        x, y = x.view(b * k, c, n), y.view(B * K, c, N)

        if n != N:
            if self.interpolation_mode is None:
                raise ValueError(
                    "Interpolation model required, but not allowed in this instance."
                )

            x = interpolate(x, N, mode=self.interpolation_mode)  # [B, k, c, N]

        # if proj_query is None or proj_key is None or proj_value is None:
        #     proj_query, proj_key, proj_value = self.extract_qkv(x, y)

        #     # [B X k X (C/r)*n_heads X N], [B X k X (C/r)*n_heads X N], [B X k X (OC)*n_heads X N]
        #     proj_query, proj_key, proj_value = (
        #         proj_query.view(B, k, self.out_channels, N),
        #         proj_key.view(B, k, self.out_channels, N),
        #         proj_value.view(B, K, self.y_out_channels, N),
        #     )
        if proj_query is None:
            proj_query = self.query(x)  # [B*k X (C*n_heads | OC*n_heads) X N]
            proj_query = proj_query.view(
                B, k, self.out_channels, N
            )  # [B X k X (C/r)*n_heads X N]
        if proj_key is None:
            proj_key = self.key(y)  # [B*k X (C*n_heads | OC*n_heads) X N]
            proj_key = proj_key.view(
                B, k, self.out_channels, N
            )  # [B X k X (C/r)*n_heads X N]
        if proj_value is None:
            proj_value = self.value(y)  # [B*k X (C*n_heads | OC*n_heads) X N]
            proj_value = proj_value.view(
                B, K, self.y_out_channels, N
            )  # [B X k X (OC)*n_heads X N]

        # [B X k X N X (C/r)*n_heads], [B X k X N X (C/r)*n_heads], [B X K X N X (OC)*n_heads]
        proj_query, proj_key, proj_value = (
            proj_query.transpose(2, 3),
            proj_key.transpose(2, 3),
            proj_value.transpose(2, 3),
        )

        if self.n_heads != 1:
            split_size = [
                self.out_channels // self.n_heads,
                self.out_channels // self.n_heads,
                self.y_out_channels // self.n_heads,
            ]
            # n_heads X B X k_q X N X (C/r), n_heads X B X k_k X N X (C/r), n_heads X B X k_v == k_k X N X OC
            proj_query, proj_key, proj_value = (
                split_cat(proj_query, split_size[0], -1, -1),
                split_cat(proj_key, split_size[1], -1, -1),
                split_cat(proj_value, split_size[2], -1, -1),
            )

        # TODO@ShivamPR21: Added new level of attention, check for integrity
        if self.xa == 0:
            st_dim, end_dim = 1, 2
            if k != K:
                raise ValueError(
                    "For no cross attention, the elements in x and y should be same s.t. k == K"
                )
            # n_heads X B*k_q X N X (C/r), n_heads X B*k_k X N X (C/r), n_heads X B*(k_v == k_k) X N X OC
            proj_query, proj_key, proj_value = (
                proj_query.flatten(st_dim, end_dim),
                proj_key.flatten(st_dim, end_dim),
                proj_value.flatten(st_dim, end_dim),
            )
            if mask is not None and (
                mask.shape[-2:] != [N, N] or mask.shape[:] != [self.n_heads, N, N]
            ):
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )
        elif self.xa == 1:
            st_dim, end_dim = 2, 3
            # n_heads X B X (k_q*N) X (C/r), n_heads X B X (k_k*N) X (C/r), n_heads X B X ((k_v == k_k)*N) X OC
            proj_query, proj_key, proj_value = (
                proj_query.flatten(st_dim, end_dim),
                proj_key.flatten(st_dim, end_dim),
                proj_value.flatten(st_dim, end_dim),
            )
            if mask is not None and (
                mask.shape[-2:] != [k * N, K * N]
                or mask.shape[:] != [self.n_heads, k * N, K * N]
            ):
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )
        else:
            st_dim, end_dim = 1, 2
            # n_heads X B*N X (k_q) X (C/r), n_heads X B*N X (k_k) X (C/r), n_heads X B*N X (k_v == k_k) X OC
            proj_query, proj_key, proj_value = (
                proj_query.transpose(3, 2).flatten(st_dim, end_dim),
                proj_key.transpose(3, 2).flatten(st_dim, end_dim),
                proj_value.transpose(3, 2).flatten(st_dim, end_dim),
            )
            if mask is not None and (
                mask.shape[-2:] != [k, K] or mask.shape[:] != [self.n_heads, k, K]
            ):
                raise ValueError(
                    "Check for correct mask size, according to the `attention_type` requested."
                )

        # [n_heads, B, C', (k_k * N)] or [n_heads, B*k_k, C', (N)] or [n_heads, B*N, C', (k_k)]
        proj_key = proj_key.transpose(3, 2)

        proj_query, proj_key, proj_value = (
            proj_query.flatten(0, 1),  # n_heads * B X ...
            proj_key.flatten(0, 1),  # n_heads * B X ...
            proj_value.flatten(0, 1),  # n_heads * B X ...
        )

        # n_heads X B X (k_q * N) X (k_k * N) or n_heads X B*(k_q == k_k) X (N) X (N) or n_heads X B*N X (k_q) X (k_k)
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        # Mask out unwanted attentions
        if mask is not None:
            # mask is of shape =>
            # (k_q * N) X (k_k * N) or (N) X (N) or (k_q) X (k_k)
            energy[..., ~mask] = -torch.inf  # TODO@ShivamPR21: Check for integrity

        # n_heads X B X (k_q * N) X (k_k * N) or n_heads X B*(k_q == k_k) X (N) X (N) or n_heads X B*N X (k_q) X (k_k)
        attention = self.softmax(energy)

        # [n_heads, B, (k_q * N), OC] or [n_heads, B*k_k, N, OC] or [n_heads, B*N, (k_q), OC]
        out = torch.bmm(attention, proj_value)

        # Final output reshape
        # [n_heads X B X k X N X OC]
        if self.xa == 2:
            out = out.view(
                self.n_heads, -1, N, k, self.y_out_channels // self.n_heads
            ).transpose(
                3, 2
            )  # [n_heads X B X N X k_q X OC] => [n_heads X B X k_q X N X OC]
        else:
            out = out.view(
                self.n_heads, -1, k, N, self.y_out_channels // self.n_heads
            )  # [n_heads X B X k_q X N X OC]

        # out = split_cat(out, B, 0, -1)  # [n_heads X B X k_q X N X OC]

        if self.residual:
            if self.projection is not None:
                x = (
                    self.projection(x).view(B, k, -1, N).transpose(2, 1)
                )  # B X k_q X N X OC

            x = x.unsqueeze(dim=0).repeat(
                self.n_heads, 1, 1, 1, 1
            )  # [n_head X B X k_q X N X OC]

            if self.gamma is None:
                raise ValueError(
                    "Trying to use gamma variable, which is not defined for this instance."
                )

            out = (self.gamma * out + x) / (1 + self.gamma)

            return out

        out = out.permute(1, 0, 3, 2, 4)  # [B X n_heads X K X OC X N]

        if final_squeeze:
            out = out.squeeze(dim=2)  # [B X n_heads X OC X N]
            if self.n_heads == 1:
                out = out.squeeze(dim=1)  # [B X OC X N]

        return out

    def shape(self, in_shape: int, y_in_shape: int):
        return y_in_shape


class MultiHeadSelfAttention2d(MultiHeadAttention2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        n_heads: int = 1,
        residual: bool = True,
        kernel_size: int | Tuple[int, int] = 1,
        attention_type: int = 0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            None,
            None,
            n_heads,
            residual,
            kernel_size,
            None,
            attention_type,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return super().forward(x, x, mask=mask)


class MultiHeadSelfAttention1d(MultiHeadAttention1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        n_heads: int = 1,
        residual: bool = True,
        kernel_size: int = 1,
        attention_type: int = 0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            None,
            None,
            n_heads,
            residual,
            kernel_size,
            None,
            attention_type,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return super().forward(x, x, mask=mask)
