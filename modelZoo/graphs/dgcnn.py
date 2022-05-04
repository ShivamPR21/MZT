from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from moduleZoo.graphs import GraphConv2d


class DGCNN(nn.Module):

    def __init__(self, k: int,
                 embed_dim: int = 512,
                 cfg: Optional[List[Tuple[int, int, int, bool]]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        self.cfg = [(3, 32, 1, False),
                    (32, 64, 1, False),
                    (64, 128, 1, True),
                    (128, 256, 1, False),
                    (256, 512, 1, False)] if cfg is None else cfg
        self.activation_layer = nn.SELU if activation_layer is None else activation_layer

        self.layers = nn.ModuleList()
        cat_dim = 0
        for cfg in self.cfg:
            self.layers.append(GraphConv2d(*cfg[:-1], k,
                                           norm_layer=nn.BatchNorm2d if cfg[-1] else None,
                                           activation_layer=self.activation_layer))
            cat_dim += cfg[1]

        self.final_conv = GraphConv2d(cat_dim,
                                       embed_dim,
                                       1,
                                       activation_layer=self.activation_layer)

    def forward(self, x: torch.Tensor):
        # Assumed shape x: [B, d, n]
        out_lst: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            out_lst.append(x)

        x = torch.cat(tuple(out_lst), dim=1)

        x = self.final_conv(x)
        return x
