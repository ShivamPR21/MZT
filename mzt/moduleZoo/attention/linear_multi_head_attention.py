from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import split_cat


class MultiHeadAttentionLinear(nn.Module):
    """ Multi Head Self attention Layer."""
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 y_in_dim: Optional[int] = None,
                 y_out_dim: Optional[int] = None,
                 n_heads: int = 1,
                 residual: bool = True,
                 interpolation_mode: Optional[str] = 'nearest'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        self.y_in_dim = self.in_dim if y_in_dim is None else y_in_dim
        self.y_out_dim = self.out_dim if y_out_dim is None else y_out_dim

        self.n_heads = n_heads
        self.residual = residual
        self.interpolation_mode = interpolation_mode

        self.out_dim *= n_heads
        self.y_out_dim *= n_heads

        self.query = nn.Linear(self.in_dim, self.out_dim)
        self.key = nn.Linear(self.in_dim, self.out_dim)

        self.value = nn.Linear(self.y_in_dim, self.y_out_dim)

        # gamma as the shape of expanded dims with n_heads, so [n_heads, 1, _, _, ...]
        self.gamma = nn.Parameter(torch.rand((n_heads, 1, 1) if n_heads == 1 else (n_heads, 1, 1, 1))+0.001) if self.residual else None

        self.proj = 'id' if self.y_out_dim//self.n_heads == self.y_in_dim and not self.residual else 'projection'
        self.projection = nn.Linear(self.y_in_dim, self.y_out_dim//self.n_heads, bias=False) if self.proj == 'projection' else None

        self.softmax = nn.Softmax(dim=-1)

    def extract_qkv(self, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.query(x), self.key(y), self.value(y)
        return q, k, v

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                proj_query: torch.Tensor | None = None,
                proj_key: torch.Tensor | None = None,
                proj_value: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # trunk-ignore(ruff/D401)
        """
            inputs :
                x : input feature maps( B X k_q X N) or ( B X N)
                y : feature map attention to be applied( B X k_v X ON) or ( B X ON)
            returns :
                out : self attention value or + input feature..
        """
        final_squeeze = False

        if x.ndim == 2:
            x = x.unsqueeze(dim=1)
            final_squeeze = True

        if y.ndim == 2:
            y = y.unsqueeze(dim=1)
            final_squeeze = True

        if not (x.ndim == 3 and y.ndim == 3):
            raise ValueError(f'Got {x.ndim = } and {y.ndim = } should be 3')

        b, k, n = x.size()
        B, K, N = y.size()

        if not (b == B):
            raise ValueError(f'Batch size of both input doesn\'t match, given {b = } and {B = }')

        if (proj_query is None or proj_key is None or proj_value is None):
            # B X k_q X N * n_heads # B X k_k X N * n_heads # B X k_v == k_k X ON * n_heads
            proj_query, proj_key, proj_value = self.extract_qkv(x, y)

        if self.n_heads != 1:
            split_size = [self.out_dim//self.n_heads, self.out_dim//self.n_heads, self.y_out_dim//self.n_heads]
            # n_heads*B X k_k X N, n_heads*B X k_q X N, n_heads*B X k_v == k_k X ON
            proj_query, proj_key, proj_value = \
                split_cat(proj_query, split_size[0], 2, 0), \
                    split_cat(proj_key, split_size[1], 2, 0), \
                        split_cat(proj_value, split_size[2], 2, 0)

        proj_key = proj_key.transpose(2, 1) # n_heads*B X N X k_k

        energy = torch.bmm(proj_query,proj_key) # transpose check # n_heads*B X k_q X k_k

        # Mask out unwanted attentions
        if mask is not None:
            energy[..., ~mask] = -torch.inf

        attention = self.softmax(energy) # n_heads*B X k_q X k_k

        out = torch.bmm(attention.transpose(2, 1), proj_value) # n_heads*B X k_q X ON

        out = split_cat(out, B, 0, -1) # n_heads X B X k_q X ON

        if self.residual:
            if self.projection is not None:
                x = self.projection(x) # B X k_q X ON

            x = x.unsqueeze(dim=0).repeat(self.n_heads, 1, 1, 1) # n_head X B X k_q X ON

            if self.gamma is None:
                raise ValueError('Trying to use gamma variable, which is not defined for this instance.')

            out = (self.gamma*out + x)/(self.gamma + 1.)

        out = out.permute(1, 0, 2, 3) # B X n_heads X k_q X ON

        if final_squeeze:
            out = out.squeeze(dim=2) # B X n_heads X ON
            if self.n_heads == 1:
                out = out.squeeze(dim=1) # B X ON

        return out

    def shape(self, in_shape: int, y_in_shape: int) -> int | List[int]:
        return self.y_out_dim//self.n_heads

class MultiHeadSelfAttentionLinear(MultiHeadAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None, n_heads: int = 1, residual: bool = True):
        super().__init__(in_dim, out_dim, None, None, n_heads, residual, None)

    def forward(self, x: torch.Tensor, proj_query: torch.Tensor | None = None,
                proj_key: torch.Tensor | None = None, proj_value: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x, x, proj_query, proj_key, proj_value, mask)

class SelfAttentionLinear(MultiHeadSelfAttentionLinear):

    def __init__(self, in_dim: int, out_dim: int | None = None, residual: bool = True):
        super().__init__(in_dim, out_dim, 1, residual)

    def forward(self, x: torch.Tensor, proj_query: torch.Tensor | None = None,
                proj_key: torch.Tensor | None = None, proj_value: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x, proj_query, proj_key, proj_value, mask)
