from typing import Callable, Optional

import torch
import torch.nn as nn

from ..convolution import ConvNormActivation2d
from .utils import knn_features


def get_graph_features(x: torch.Tensor, idx: Optional[torch.Tensor] = None, k: Optional[int] = None, mode: str = 'local+global') -> torch.Tensor:
    B, n, d = x.size() # [B, n, d]

    features = knn_features(x, idx, k) # [B, n, k, d]
    x = x.view(B, n, 1, d).repeat(1, 1, k, 1) # [B, n, k, d]

    if mode == 'local+global':
        features = torch.cat((features-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # [B, 2*d, n, k]
    elif mode == 'local':
        features = (features-x).permute(0, 3, 1, 2).contiguous() # [B, d, n, k]
    elif mode == 'global':
        features = features.permute(0, 3, 1, 2).contiguous() # [B, d, n, k]
    else:
        raise NotImplementedError

    return features

class GraphConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels : int,
                 kernel_size: int = 1,
                 k: int = 10,
                 reduction: str = 'max',
                 features: str = 'local+global',
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k = k
        self.reduction = reduction
        self.features = features

        if self.features == 'local+global':
            self.in_channels *= 2

        self.conv = ConvNormActivation2d(self.in_channels,
                                         self.out_channels,
                                         (1, kernel_size),
                                         padding=0,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, d, n = x.size()
        x = x.transpose(2, 1) # [B, n, d]

        x = get_graph_features(x, None, self.k, self.features) # [B, 2*d, n, k]

        x = self.conv(x)

        if self.reduction == 'max':
            x = x.max(dim=-1)[0] # [B, d, n]
        elif self.reduction == 'mean':
            x = x.mean(dim=-1) # [B, d, n]
        else:
            raise NotImplementedError

        return x

    def shape(self, in_shape:int):
        return in_shape
