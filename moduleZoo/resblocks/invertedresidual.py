from typing import Callable, Tuple

import torch
import torch.nn as nn

from ..convolution import (ConvInvertedBlock1d, ConvInvertedBlock2d,
                           ConvNormActivation1d, ConvNormActivation2d)


class ConvInvertedResidualBlock2d(ConvInvertedBlock2d):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: float,
                 out_channels: float | None = None,
                 kernel_size: int | Tuple[int, int] = 3,
                 stride: int | Tuple[int, int] = 1,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
                 channel_shuffle: bool = False,
                 grouping: int = 1) -> None:
        super().__init__(in_channels,
                         expansion_ratio,
                         out_channels,
                         kernel_size,
                         stride,
                         norm_layer,
                         activation_layer,
                         channel_shuffle,
                         grouping)

        self.proj_type = 'id' if stride == 1 and self.out_channels == self.in_channels else 'projection'

        self.projection = ConvNormActivation2d(self.in_channels,
                                                self.out_channels,
                                                1,
                                                stride,
                                                padding='stride_effective',
                                                bias=False,
                                                norm_layer=None,
                                                activation_layer=None) if self.proj_type == 'projection' else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)

        if self.channel_shuffle is not None:
            x_ = self.channel_shuffle(x_)

        x_ = self.conv2(x_)
        x_ = self.conv3(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
        else:
            x = x + x_

        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: Tuple[int, int]):
        final_conv_shape = super().shape(in_shape)
        final_proj_shape = self.projection.shape(in_shape) if self.projection is not None else in_shape
        assert(final_conv_shape == final_proj_shape)

        return final_conv_shape


class ConvInvertedResidualBlock1d(ConvInvertedBlock1d):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: float,
                 kernel_size: int | Tuple[int, int] = 3,
                 stride: int | Tuple[int, int] = 1,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
                 channel_shuffle: bool = False,
                 grouping: int = 1) -> None:
        super().__init__(in_channels,
                         expansion_ratio,
                         kernel_size,
                         stride,
                         norm_layer,
                         activation_layer,
                         channel_shuffle,
                         grouping)

        self.proj_type = 'id' if stride == 1 else 'projection'

        self.projection = ConvNormActivation1d(self.in_channels,
                                                self.out_channels,
                                                1,
                                                stride,
                                                padding='stride_effective',
                                                bias=False,
                                                norm_layer=None,
                                                activation_layer=None) if self.proj_type == 'projection' else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)

        if self.channel_shuffle is not None:
            x_ = self.channel_shuffle(x_)

        x_ = self.conv2(x_)
        x_ = self.conv3(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
            x = self.activation(x) if self.activation is not None else x
            return x

        x = x + x_
        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: Tuple[int, int]):
        final_conv_shape = super().shape(in_shape)
        final_proj_shape = self.projection.shape(in_shape) if self.projection is not None else in_shape
        assert(final_conv_shape == final_proj_shape)

        return final_conv_shape
