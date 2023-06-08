from typing import Optional

import torch


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    # Assumed shape of x -> [B, n, d]
    x = x.unsqueeze(dim=1) # [B, 1, n, d]
    n_x = x.transpose(2, 1) # [B, n, 1, d]

    x = -torch.sum((x - n_x)**2, dim=-1) # [B, n, n]
    idx = x.topk(k, dim=-1, sorted=True)[1] # [B, n, k], sorted indexes

    return idx

def knn_features(x: torch.Tensor, idx: Optional[torch.Tensor] = None, k: Optional[int] = None) -> torch.Tensor:
    assert(idx is not None or k is not None)

    B, n, d = x.size()

    if idx is None:
        idx = knn(x, k) # [B, n, k]

    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1)*n
    idx = (idx + idx_base).view(-1) # [B*n*k]

    if not x.is_contiguous():
        x = x.contiguous() # [B, n, d], make contiguous

    features = x.view(B*n, d)[idx, :] # [B*n*k, d]
    features = features.view(B, n, k, d) # [B, n, k, d]

    return features
