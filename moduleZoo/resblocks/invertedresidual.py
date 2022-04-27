'''
Copyright (C) 2021  Shivam Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..convolution import (
    ConvInvertedBlock1d,
    ConvInvertedBlock2d,
    ConvNormActivation1d,
    ConvNormActivation2d,
)


class ConvInvertedResidualBlock2d(ConvInvertedBlock2d):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: float,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6,
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

        self.projection = ConvNormActivation2d(in_channels,
                                                in_channels,
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


class ConvInvertedResidualBlock1d(ConvInvertedBlock1d):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: float,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6,
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

        self.projection = ConvNormActivation1d(in_channels,
                                                in_channels,
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
