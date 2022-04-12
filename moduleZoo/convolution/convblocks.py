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

from typing import Callable, List, Optional, Tuple

import torch.nn as nn


class Conv2DNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        transposed: bool = False,
        output_padding: Tuple[int, int] = (0, 0),
        inplace: bool = True,
    ) -> None:
        """Typical Convolution Normalization and Activation stack for easier implementation

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
            stride (int, optional): Stride value for convolution. Defaults to 1.
            padding (Optional[int], optional): Padding value, (passed to nn.Conv2D). Defaults to None.
            groups (int, optional): Number of groups input tensor will be divided, (passed to nn.Conv2D). Defaults to 1.
            norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation_layer (Optional[Callable[..., nn.Module]], optional): Activation function. Defaults to nn.ReLU.
            dilation (int, optional): Dilation used in convolution (passed in nn.Conv2D). Defaults to 1.
            inplace (bool, optional): Whether to use inplace operations or not. Defaults to True.
        """
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers:List[nn.Module] = None

        if transposed:
            layers = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                    bias=norm_layer is None,
                )
            ]
        else:
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    groups=groups,
                    bias=norm_layer is None,
                )
            ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels
