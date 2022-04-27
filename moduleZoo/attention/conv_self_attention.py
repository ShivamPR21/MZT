from typing import Tuple, Union

from . import MultiHeadSelfAttention1d, MultiHeadSelfAttention2d


class SelfAttention2d(MultiHeadSelfAttention2d):

    def __init__(self, in_channels: int, out_channels: int, reduction_factor: int = 8, residual: bool = True, kernel_size: Union[int, Tuple[int, int]] = 1):
        super().__init__(in_channels, out_channels, 1, reduction_factor, residual, kernel_size)


class SelfAttention1d(MultiHeadSelfAttention1d):

    def __init__(self, in_channels: int, out_channels: int, reduction_factor: int = 8, residual: bool = True, kernel_size: Union[int, Tuple[int, int]] = 1):
        super().__init__(in_channels, out_channels, 1, reduction_factor, residual, kernel_size)
