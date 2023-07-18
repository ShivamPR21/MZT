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

from typing import Any, Callable, Dict, List, Tuple, Type

import torch
import torch.nn as nn

from ..utils import _pair


class ConvNormActivation2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] | str = 0,
        groups: int = 1,
        dilation: int | Tuple[int, int] = 1,
        bias: bool = True,
        transposed: bool = False,
        output_padding: int | Tuple[int, int] = 0,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
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
        super().__init__()
        if padding == 'stride_effective':
            padding = (kernel_size - 1) // 2 * dilation

        # TODO@ShivamPR21: #7 Provide onnx support for `padding=same`

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        self.cfg: Dict[str, Any] = {'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'output_padding': output_padding,
                    'dilation': dilation,
                    'groups': groups,
                    'bias': bias,
                    'transposed': transposed}

        self.conv: Type[nn.Module] = None

        if transposed:
            self.conv = nn.ConvTranspose2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            output_padding=output_padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        )
        else:
            self.conv = nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        )

        self.norm = None

        if norm_layer is not None:
            try:
                self.norm = norm_layer(out_channels, affine=True)
            except TypeError:
                self.norm = norm_layer(out_channels)

        self.act = activation_layer() if activation_layer is not None else None

        self.cfg.update({'dim_cnst0': 0 if padding == 'same' else (2 * padding[0] - dilation[0] * ( kernel_size[0] - 1 ) - 1),
                         'dim_cnst1': 0 if padding == 'same' else (2 * padding[1] - dilation[1] * ( kernel_size[1] - 1 ) - 1)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.act(x) if self.act is not None else x

        return x

    def shape(self, in_shape: Tuple[int, int]):
        H, W = in_shape
        H_out, W_out = None, None

        if not self.cfg['transposed']:
            H_out = (H + self.cfg['dim_cnst0'])//self.cfg['stride'][0]+1
            W_out = (W + self.cfg['dim_cnst1'])//self.cfg['stride'][1]+1
        else:
            H_out = (H-1) * self.cfg['stride'][0] - self.cfg['dim_cnst0'] + self.cfg['output_padding'][0] + 1
            W_out = (W-1) * self.cfg['stride'][1] - self.cfg['dim_cnst1'] + self.cfg['output_padding'][1] + 1

        return H_out, W_out


class ConvNormActivation1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
        transposed: bool = False,
        output_padding: int = 0,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
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
        if padding == 'stride_effective':
            padding = (kernel_size - 1) // 2 * dilation

        # TODO@ShivamPR21: #7 Provide onnx support for `padding=same`

        self.cfg: Dict[str, Any] = {'in_channels': in_channels,
                                    'out_channels': out_channels,
                                    'kernel_size': kernel_size,
                                    'stride': stride,
                                    'output_padding': output_padding,
                                    'dilation': dilation,
                                    'groups': groups,
                                    'bias': bias,
                                    'transposed': transposed}

        layers:List[nn.Module] = None

        if transposed:
            layers = [
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            ]
        else:
            layers = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.cfg.update({'dim_cnst': 0 if padding == 'same' else (2 * padding - dilation * ( kernel_size - 1 ) - 1)})

    def shape(self, in_shape: int):
        L = in_shape
        L_out = None

        if not self.cfg['transposed']:
            L_out = (L + self.cfg['dim_cnst'])//self.cfg['stride']+1
        else:
            L_out = (L-1) * self.cfg['stride'] - self.cfg['dim_cnst'] + self.cfg['output_padding'] + 1

        return L_out
