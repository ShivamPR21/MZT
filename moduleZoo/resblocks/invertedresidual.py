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

import torch.nn as nn
from torch import Tensor

from ..convolution import Conv2DNormActivation


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_ratio: float,
                 stride: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        """Inverted Residual Block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            expansion_ratio (float): Expantion ratio, determines the width of hidden layers
            stride (int): stride for conv blocks, must be in [1, 2], if >1 then rsidual connection will also be skipped
            norm_layer (Optional[Callable[..., nn.Module]], optional): normalization layer, applied just after convolition and before activation function. Defaults to None.
        """
        super().__init__()
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_channels = round(in_channels * expansion_ratio)
        self.res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expansion_ratio != 1:
            layers.append(Conv2DNormActivation(in_channels, hidden_channels, kernel_size=1,
                                             stride=stride, norm_layer=norm_layer, activation_layer=nn.Relu6))
        layers.extend(
            [
                Conv2DNormActivation(
                    hidden_channels,
                    hidden_channels,
                    stride = stride,
                    groups = hidden_channels,
                    norm_layer = norm_layer,
                    activation_layer = nn.ReLU6
                ),

                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                norm_layer(out_channels)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
