from typing import Callable, Tuple

import torch
import torch.nn as nn

from ..convolution import ConvNormActivation1d, ConvNormActivation2d


class ConvBottleNeckResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int,
        out_channels: int | None = None,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.activation = activation_layer() if activation_layer is not None else None

        self.hidden_channels = in_channels // reduction_ratio
        if not (self.hidden_channels != 0):
            raise  # TODO@ShivamPR21: Provide better debug argument

        self.out_channels = in_channels if out_channels is None else out_channels

        self.proj_type = (
            "id" if stride == 1 and in_channels == out_channels else "projection"
        )

        self.conv1 = ConvNormActivation2d(
            self.in_channels,
            self.hidden_channels,
            1,
            stride,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.conv2 = ConvNormActivation2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.conv3 = ConvNormActivation2d(
            self.hidden_channels,
            self.out_channels,
            1,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.projection = (
            ConvNormActivation2d(
                self.in_channels,
                self.out_channels,
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
        x_ = self.conv3(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
            x = self.activation(x) if self.activation is not None else x
            return x

        x = x + x_
        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: Tuple[int, int]):
        shape1 = self.conv1.shape(in_shape)
        shape2 = self.conv2.shape(shape1)
        final_conv_shape = self.conv3.shape(shape2)
        final_proj_shape = (
            self.projection.shape(in_shape) if self.projection is not None else in_shape
        )
        if not (final_conv_shape == final_proj_shape):
            raise  # TODO@ShivamPR21: Provide better debug info

        return final_conv_shape


class ConvBottleNeckResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.activation = activation_layer() if activation_layer is not None else None

        self.hidden_channels = in_channels // reduction_ratio
        if not (self.hidden_channels != 0):
            raise  # TODO@ShivamPR21: Provide better debug info

        self.out_channels = in_channels if out_channels is None else out_channels

        self.proj_type = (
            "id" if stride == 1 and in_channels == out_channels else "projection"
        )

        self.conv1 = ConvNormActivation1d(
            self.in_channels,
            self.hidden_channels,
            1,
            stride,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.conv2 = ConvNormActivation1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.conv3 = ConvNormActivation1d(
            self.hidden_channels,
            self.out_channels,
            1,
            padding="stride_effective",
            bias=True if norm_layer is None else False,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.projection = (
            ConvNormActivation1d(
                self.in_channels,
                self.out_channels,
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
        x_ = self.conv3(x_)

        if self.projection is not None:
            x = self.projection(x) + x_
            x = self.activation(x) if self.activation is not None else x
            return x

        x = x + x_
        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: int):
        shape1 = self.conv1.shape(in_shape)
        shape2 = self.conv2.shape(shape1)
        final_conv_shape = self.conv3.shape(shape2)
        final_proj_shape = (
            self.projection.shape(in_shape) if self.projection is not None else in_shape
        )
        if not (final_conv_shape == final_proj_shape):
            raise  # TODO@ShivamPR21: Provide better debug info

        return final_conv_shape
