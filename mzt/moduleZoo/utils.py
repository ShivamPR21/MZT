from collections.abc import Iterable
from itertools import repeat

import torch


class NestedTensor:
    def __init__(
        self,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_enc: torch.Tensor | None = None,
    ) -> None:
        self._data, self._mask, self._pos_enc = data, mask, pos_enc

    def __repr__(self) -> str:
        repr = f"(data): {self.data}"

        if self.mask is not None:
            repr += f"\n(mask): {self.mask}"

        if self.pos_enc is not None:
            repr += f"\n(pose encodings): {self.pos_enc}"

        return repr

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask

    @property
    def pos(self) -> torch.Tensor | None:
        return self._pos_enc


def with_pose(x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
    if pos is not None:
        return x + pos

    return x + pos


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

if __name__ == "__main__":
    v = 0
    print(_single(v))

    v = (0, 0)
    print(_pair(v))
