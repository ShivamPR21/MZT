from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from moduleZoo.attention.utils import split_cat
from moduleZoo.dense import LinearNormActivation

from .utils import knn_features


def get_graph_features(x: torch.Tensor, idx: torch.Tensor | None = None, k: int | None = None, mode: str = 'local+global') -> torch.Tensor:
    B, n, d = x.size() # [B, n, d]

    features = knn_features(x, idx, k) # [B, n, k, d]
    x = x.view(B, n, 1, d).repeat(1, 1, k, 1) # [B, n, k, d]

    if mode == 'local+global':
        features = torch.cat((features-x, x), dim=3).contiguous() # [B, n, k, 2*d]
    elif mode == 'local':
        features = (features-x).contiguous() # [B, n, k, d]
    elif mode == 'global':
        features = features.contiguous() # [B, n, k, d]
    else:
        raise NotImplementedError

    return features

class GraphConv(LinearNormActivation):

    def __init__(self,
                 in_channels: int,
                 out_channels : int,
                 bias: bool = True,
                 k: int = 10,
                 reduction: str = 'max',
                 features: str = 'local+global',
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None,
                 dynamic_batching: bool = False,
                 enable_offloading: bool = True) -> None:

        if features == 'local+global':
            in_channels *= 2

        super().__init__(in_channels, out_channels, bias, norm_layer, activation_layer)

        self.k = k
        self.reduction = reduction
        self.features = features
        self.db = dynamic_batching
        self.matrix_op_device = torch.device('cpu') if enable_offloading else None

    def static_forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, n, d = x.size()
        ### Wrap for cpu offloading ###
        ###############################
        source_device = x.device

        if self.matrix_op_device is not None and source_device != self.matrix_op_device:
            x = x.to(self.matrix_op_device)

        x = get_graph_features(x, None, self.k, self.features) # [B, n, k, 2*d]

        if source_device != x.device:
            x = x.to(source_device)

        ### Wrap for cpu offloading ###
        ###############################

        x = super().forward(x)

        if self.reduction == 'max':
            x = x.max(dim=2)[0] # [B, n, d']
        elif self.reduction == 'mean':
            x = x.mean(dim=2) # [B, n, d']
        else:
            raise NotImplementedError

        return x

    def dynamic_static_forward(self, x: torch.Tensor, node_group_sizes: np.ndarray | List[int]) -> torch.Tensor:
        # x -> [n, d]
        # batch_sizes -> [n......]
        N, d = x.shape

        if not isinstance(node_group_sizes, np.ndarray):
            node_group_sizes = np.array(node_group_sizes, dtype=np.uint32)

        result = torch.zeros((x.shape[0], self.shape()), device=x.device)

        sz_arr = np.array([_.numpy() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(node_group_sizes.tolist(), dim=0)], dtype=object)

        for unl in np.unique(node_group_sizes):
            idx_map = (node_group_sizes == unl)
            idxs = np.concatenate(sz_arr[idx_map], axis = 0).astype(np.uint32).tolist()

            x_ = x[idxs, :] # [n_, d]
            b = x_.shape[0]//unl
            x_ = x_.view((b, unl, d)) # [N//b, b, d]
            x_ = self.static_forward(x_)
            result[idxs, :] = x_.view((b*unl, self.shape()))

            assert(torch.all(x_ == split_cat(result[idxs, :], int(unl), 0, -1)).item())

        return result

    def dynamic_forward(self, x: torch.Tensor, node_group_sizes: np.ndarray | List[int]) -> torch.Tensor:
        # x -> [n, d]
        # node_group_sizes -> [n........]
        if isinstance(node_group_sizes, np.ndarray):
            node_group_sizes = node_group_sizes.tolist()

        ### Wrap for cpu offloading ###
        ###############################
        source_device = x.device

        if self.matrix_op_device is not None and source_device != self.matrix_op_device:
            x = x.to(self.matrix_op_device)

        x = torch.cat(
            [get_graph_features(x_, None, self.k, self.features) for x_ in x.unsqueeze(dim=0).split(node_group_sizes, dim=1)],
        dim=0).squeeze(dim=0) # [N, k, d]

        if source_device != x.device:
            x = x.to(source_device)

        ### Wrap for cpu offloading ###
        ###############################

        x = super().forward(x) # [N, k, d']

        if self.reduction == 'max':
            x = x.max(dim=2)[0] # [B, n, d']
        elif self.reduction == 'mean':
            x = x.mean(dim=2) # [B, n, d']
        else:
            raise NotImplementedError

        return x

    def forward(self, x: torch.Tensor, batch_sizes: np.ndarray | List[int] | None = None) -> torch.Tensor:
        if batch_sizes is not None:
            if self.db:
                print('Using static dynamic batching.')
                x = self.dynamic_static_forward(x, batch_sizes)
            else:
                print('Using complete dynamic batching')
                x = self.dynamic_forward(x, batch_sizes)
        else:
            print('Using static forward')
            x = self.static_forward(x)

        return x

    def shape(self, in_shape: int | None = None):
        return super().shape(in_shape)
