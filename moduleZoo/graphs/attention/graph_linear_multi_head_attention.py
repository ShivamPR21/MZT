from typing import List

import numpy as np
import torch
from moduleZoo.attention import MultiHeadAttentionLinear


class MultiHeadGraphAttentionLinear(MultiHeadAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None, y_in_dim: int | None = None, y_out_dim: int | None = None, n_heads: int = 1,
                 residual: bool = True, interpolation_mode: str | None = 'nearest', dynamic_batching : bool = False):
        super().__init__(in_dim, out_dim, y_in_dim, y_out_dim, n_heads, residual, interpolation_mode)
        self.db = dynamic_batching # Allows dynamic batching for efficient compute

    def forward(self, x: torch.Tensor, y: torch.Tensor, node_group_sizes: np.ndarray | List[int]) -> torch.Tensor:
        # x -> [N, in_dim]
        # y -> [N, in_dim]
        # node_group_sizes -> [n.......]


        result : torch.Tensor | List[torch.Tensor] = []

        if self.db:
            if not isinstance(node_group_sizes, np.ndarray):
                node_group_sizes = np.array(node_group_sizes, dtype=np.uint32)

            result = torch.zeros((x.shape[0], self.n_heads, self.shape(x.shape[-1], y.shape[-1])), device=x.device)

            sz_arr = np.array([_.numpy() for _ in torch.arange(0, x.shape[0], dtype=torch.int32).split(node_group_sizes.tolist(), dim=0)], dtype=object)

            for uln in np.unique(node_group_sizes):
                idx_map = (node_group_sizes == uln)
                idxs = np.concatenate(sz_arr[idx_map], axis=0).tolist()

                x_, y_ = x[idxs, :], y[idxs, :]
                b = x_.shape[0]//uln
                x_, y_ = x_.view((b, uln, x_.shape[1])), y_.view((b, uln, y_.shape[1]))

                res = super().forward(x_, y_).permute(0, 2, 1, 3).flatten(0, 1)
                result[idxs, :, :] = res
        else:
            q, k, v = super().extract_qkv(x, y) # Extract q, k, v from base attention class

            if isinstance(node_group_sizes, np.ndarray):
                node_group_sizes = node_group_sizes.tolist()

            # slice according to node_group_sizes
            x = x.unsqueeze(dim=0).split(node_group_sizes, dim=1)
            y = y.unsqueeze(dim=0).split(node_group_sizes, dim=1)
            q = q.unsqueeze(dim=0).split(node_group_sizes, dim=1)
            k = k.unsqueeze(dim=0).split(node_group_sizes, dim=1)
            v = v.unsqueeze(dim=0).split(node_group_sizes, dim=1)

            for i in range(len(node_group_sizes)):
                result.append(
                    super().forward(x[i], y[i], q[i], k[i], v[i]).permute(0, 2, 1, 3).flatten(0, 1)
                ) # [k, n_heads, ON]

            result = torch.cat(result, dim=0) # [N, n_heads, ON]

        if self.n_heads == 1:
            result = result.squeeze(dim=1)

        return result

    def shape(self, in_shape: int, y_in_shape: int) -> int | List[int]:
        return super().shape(in_shape, y_in_shape)

class MultiHeadSelfGraphAttentionLinear(MultiHeadGraphAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None, n_heads: int = 1,
                 residual: bool = True, interpolation_mode: str | None = 'nearest', dynamic_batching : bool = False):
        super().__init__(in_dim, out_dim, in_dim, out_dim, n_heads, residual, interpolation_mode, dynamic_batching)

    def forward(self, x: torch.Tensor, node_group_sizes: np.ndarray) -> torch.Tensor:
        return super().forward(x, x, node_group_sizes)

class SelfGraphAttentionLinear(MultiHeadSelfGraphAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None,
                 residual: bool = True, dynamic_batching : bool = False):
        super().__init__(in_dim, out_dim, 1, residual, None, dynamic_batching)

    def forward(self, x: torch.Tensor, node_group_sizes: np.ndarray) -> torch.Tensor:
        return super().forward(x, node_group_sizes)
