from typing import Optional

import torch


def knn(x: torch.Tensor, k: int, similarity: str = "euclidean") -> torch.Tensor:
    # Assumed shape of x -> [B, n, d]
    if similarity == "euclidean":
        x = x.unsqueeze(dim=1)  # [B, 1, n, d]
        n_x = x.transpose(2, 1)  # [B, n, 1, d]
        x = -torch.sum((x - n_x) ** 2, dim=-1)  # [B, n, n]
    elif similarity == "cosine":
        x = (x - x.mean(dim=1, keepdim=True)) / (x.norm(dim=-1, keepdim=True) + 1e-9)
        x = torch.bmm(x, x.transpose(1, 2))  # [B, n, n]
    else:
        raise NotImplementedError(f"Distance metric {similarity} is not implemented.")

    idx = x.topk(k, dim=-1, sorted=True).indices  # [B, n, k], sorted indexes

    return idx


def knn_features(
    x: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
    k: Optional[int] = None,
    similarity: str = "euclidean",
) -> torch.Tensor:
    if not (idx is not None) or (k is not None):
        raise  # TODO@ShivamPR21: Provide better debug argument

    B, n, d = x.size()

    if idx is None:
        idx = knn(x, k, similarity)  # [B, n, k]

    device = x.device

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * n
    idx = (idx + idx_base).view(-1)  # [B*n*k]

    if not x.is_contiguous():
        x = x.contiguous()  # [B, n, d], make contiguous

    features = x.view(B * n, d)[idx, :]  # [B*n*k, d]
    features = features.view(B, n, k, d)  # [B, n, k, d]

    return features
