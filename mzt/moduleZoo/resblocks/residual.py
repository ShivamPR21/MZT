from typing import Callable, Tuple

import torch
import torch.nn as nn

from ..convolution import ConvNormActivation1d, ConvNormActivation2d


class ConvResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
    ) -> None:
        super().__init__()

        self.proj_type = (
            "id" if stride == 1 and in_channels == out_channels else "projection"
        )

        self.activation = activation_layer() if activation_layer is not None else None

        self.conv1 = ConvNormActivation2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.conv2 = ConvNormActivation2d(
            out_channels,
            out_channels,
            kernel_size,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.projection = (
            ConvNormActivation2d(
                in_channels,
                out_channels,
                1,
                stride,
                padding="stride_effective",
                bias=False,
                norm_layer=None,
                activation_layer=None,
            )
            if self.proj_type == "projection"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)
        x_ = self.conv2(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
            x = self.activation(x) if self.activation is not None else x
            return x

        x = x + x_
        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: Tuple[int, int]):
        shape1 = self.conv1.shape(in_shape)
        final_conv_shape = self.conv2.shape(shape1)
        final_proj_shape = (
            self.projection.shape(in_shape) if self.projection is not None else in_shape
        )
        assert final_conv_shape == final_proj_shape

        return final_conv_shape


class ConvResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.proj_type = (
            "id" if stride == 1 and in_channels == out_channels else "projection"
        )

        self.activation = activation_layer() if activation_layer is not None else None

        self.conv1 = ConvNormActivation1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.conv2 = ConvNormActivation1d(
            out_channels,
            out_channels,
            kernel_size,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.projection = (
            ConvNormActivation1d(
                in_channels,
                out_channels,
                1,
                stride,
                padding="stride_effective",
                bias=False,
                norm_layer=None,
                activation_layer=None,
            )
            if self.proj_type == "projection"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.conv1(x)
        x_ = self.conv2(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
            x = self.activation(x) if self.activation is not None else x
            return x

        x = x + x_
        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: int):
        shape1 = self.conv1.shape(in_shape)
        final_conv_shape = self.conv2.shape(shape1)
        final_proj_shape = (
            self.projection.shape(in_shape) if self.projection is not None else in_shape
        )
        if not (final_conv_shape == final_proj_shape):
            raise  # TODO@ShivamPR21: Provide better debug info

        return final_conv_shape
