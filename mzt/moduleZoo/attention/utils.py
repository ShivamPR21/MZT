import torch


def split_cat(input: torch.Tensor, split_size:int, split_dim:int, cat_dim:int | None):
    if split_dim == -1:
        split_dim = input.ndim - 1

    if cat_dim == -1:
        split_dim += 1
        cat_dim += 1
        input = input.unsqueeze(dim=0)

    splits = torch.split(input, split_size, split_dim)
    out = torch.cat(splits, cat_dim if cat_dim is not None else 0)

    return out
