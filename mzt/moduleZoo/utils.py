from collections.abc import Iterable
from itertools import repeat


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
