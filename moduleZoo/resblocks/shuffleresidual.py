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

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from moduleZoo import ConvNormActivation
from torch import Tensor
from torch.nn import ChannelShuffle


class ShuffleInvertedResidual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_ratio: float,
                 grouping: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        assert(stride in [1, 2])

        self.res_connect = stride == 1 and in_channels == out_channels
        self.cat_connect = stride == 2

        hidden_channels = round(in_channels * expansion_ratio)
        self.padding = (kernel_size - 1) // 2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.activation = activation_layer if not activation_layer is None else nn.ReLU6

        layers: List[nn.Module] = []
        if expansion_ratio != 1:
            layers.append(ConvNormActivation(in_channels, hidden_channels, kernel_size=1,
                                             stride=stride, norm_layer=norm_layer,
                                             activation_layer=nn.Relu6, groups=grouping))

        layers.extend(
            [
                ChannelShuffle(grouping),
                ConvNormActivation(hidden_channels,
                                   hidden_channels,
                                   kernel_size=kernel_size,
                                   stride = stride,
                                   group = hidden_channels,
                                   norm_layer=norm_layer,
                                   activation_layer = self.activation
                ),
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                norm_layer(out_channels)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

        self.avgpool_layer = nn.AvgPool2d(kernel_size, stride, self.padding) if self.cat_connect else None

    def forward(self, x: Tensor) -> Tensor:
        if self.res_connect:
            return self.activation(x + self.conv(x))
        elif self.cat_connect:
            return self.activation(
                torch.cat(
                    (self.avgpool_layer(x), self.conv(x)), dim = 0
                    )
                )
        else:
            return self.activation(self.conv(x))
