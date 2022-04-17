from typing import Callable, Optional

import torch
import torch.nn as nn

from ..convolution import Conv2DNormActivation


class Conv2DResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6) -> None:
        super().__init__()

        self.proj_type = 'id' if stride == 1 and in_channels == out_channels else 'projection'

        self.activation = activation_layer

        self.conv1 = Conv2DNormActivation(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding='stride_effective',
                                          bias=True,
                                          norm_layer=None,
                                          activation_layer=activation_layer)

        self.conv2 = Conv2DNormActivation(out_channels,
                                          out_channels,
                                          kernel_size,
                                          padding='same',
                                          bias=False,
                                          norm_layer=None,
                                          activation_layer=None)

        self.projection = Conv2DNormActivation(in_channels,
                                                out_channels,
                                                1,
                                                stride,
                                                padding='stride_effective',
                                                bias=False,
                                                norm_layer=None,
                                                activation_layer=None) if self.proj_type == 'projection' else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)
        x_ = self.conv2(x_)

        if self.projection is not None:
            return self.activation(self.projection(x) + x_)

        return self.activation(x + x_)


class Conv2DInvertedResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 expansion_ratio: float,
                 kernel_size: int = 3,
                 stride: int = 1,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6) -> None:
        super().__init__()

        self.proj_type = 'id' if stride == 1 else 'projection'

        hidden_channels = round(in_channels * expansion_ratio)

        self.activation = activation_layer
        self.conv1 = Conv2DNormActivation(in_channels,
                                          hidden_channels,
                                          1,
                                          stride,
                                          padding='stride_effective',
                                          bias=True,
                                          norm_layer=None,
                                          activation_layer=activation_layer)

        self.conv2 = Conv2DNormActivation(hidden_channels,
                                          hidden_channels,
                                          kernel_size,
                                          padding='same',
                                          bias=True,
                                          norm_layer=None,
                                          activation_layer=activation_layer)

        self.conv3 = Conv2DNormActivation(hidden_channels,
                                          in_channels,
                                          1,
                                          stride=1,
                                          padding='same',
                                          bias=False,
                                          norm_layer=None,
                                          activation_layer=None)

        self.projection = Conv2DNormActivation(in_channels,
                                                in_channels,
                                                1,
                                                stride,
                                                padding='stride_effective',
                                                bias=False,
                                                norm_layer=None,
                                                activation_layer=None) if self.proj_type == 'projection' else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)
        x_ = self.conv2(x_)

        if self.projection is not None:
            return self.activation(self.projection(x) + x_)

        return self.activation(x + x_)
