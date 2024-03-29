from typing import Tuple

from . import MultiHeadSelfAttention1d, MultiHeadSelfAttention2d


class SelfAttention2d(MultiHeadSelfAttention2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        residual: bool = True,
        kernel_size: int | Tuple[int, int] = 1,
        attention_type: int = 0,
    ):
        super().__init__(
            in_channels, out_channels, 1, residual, kernel_size, attention_type
        )


class SelfAttention1d(MultiHeadSelfAttention1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        residual: bool = True,
        kernel_size: int = 1,
        attention_type: int = 0,
    ):
        super().__init__(
            in_channels, out_channels, 1, residual, kernel_size, attention_type
        )
