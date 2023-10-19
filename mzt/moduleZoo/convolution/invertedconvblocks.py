from typing import Callable, Tuple

import torch
import torch.nn as nn

from . import ConvNormActivation1d, ConvNormActivation2d


class ConvInvertedBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion_ratio: float,
        out_channels: int | None = None,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
        channel_shuffle: bool = False,
        grouping: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = round(in_channels * expansion_ratio)
        self.out_channels = in_channels if out_channels is None else out_channels

        self.activation = activation_layer() if activation_layer is not None else None

        self.conv1 = ConvNormActivation2d(
            in_channels,
            self.hidden_channels,
            1,
            stride,
            padding="stride_effective",
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.channel_shuffle = nn.ChannelShuffle(grouping) if channel_shuffle else None

        self.conv2 = ConvNormActivation2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size,
            padding="stride_effective",
            groups=self.hidden_channels,  # Depth wise convolution
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.conv3 = ConvNormActivation2d(
            self.hidden_channels,
            self.out_channels,
            1,
            stride=1,
            padding="stride_effective",
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        if self.channel_shuffle is not None:
            x = self.channel_shuffle(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: Tuple[int, int]):
        shape1 = self.conv1.shape(in_shape)
        shape2 = self.conv2.shape(shape1)
        final_conv_shape = self.conv3.shape(shape2)

        return final_conv_shape


class ConvInvertedBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion_ratio: float,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
        channel_shuffle: bool = False,
        grouping: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = round(in_channels * expansion_ratio)
        self.out_channels = in_channels if out_channels is None else out_channels

        self.activation = activation_layer() if activation_layer is not None else None

        self.conv1 = ConvNormActivation1d(
            in_channels,
            self.hidden_channels,
            1,
            stride,
            padding="stride_effective",
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        self.channel_shuffle = nn.ChannelShuffle(grouping) if channel_shuffle else None

        self.conv2 = ConvNormActivation1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size,
            padding="stride_effective",
            groups=self.hidden_channels,  # Depth wise convolution
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

        self.conv3 = ConvNormActivation1d(
            self.hidden_channels,
            self.out_channels,
            1,
            stride=1,
            padding="stride_effective",
            bias=norm_layer is None,
            norm_layer=norm_layer,
            activation_layer=None,
        )  # TODO@ShivamPR21: #8 Padding `same` applied though proxy (`stride_effective, stride=1`) for onnx support

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        if self.channel_shuffle is not None:
            x = self.channel_shuffle(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = self.activation(x) if self.activation is not None else x
        return x

    def shape(self, in_shape: int):
        shape1 = self.conv1.shape(in_shape)
        shape2 = self.conv2.shape(shape1)
        final_conv_shape = self.conv3.shape(shape2)

        return final_conv_shape
