from typing import Any, Callable, Dict

import torch
import torch.nn as nn


class LinearNormActivation(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bias: bool = True,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """_summary_

        Args:
            in_dims (int): _description_
            out_dims (int): _description_
            bias (bool, optional): _description_. Defaults to True.
            norm_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
            activation_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.cfg: Dict[str, Any] = {'in_dims': in_dims,
                                    'out_dims': out_dims,
                                    'bias': bias}

        self.linear = nn.Linear(
                in_dims,
                out_dims,
                bias=bias,
            )

        self.norm = None

        if norm_layer is not None:
            if norm_layer == nn.LayerNorm:
                self.norm = norm_layer(out_dims, elementwise_affine=True)
            else:
                self.norm = norm_layer(out_dims, affine=True)

        self.act = activation_layer() if activation_layer is not None else None

    def forward(self, input):
        input_dims = input.size()
        out_dims = (input_dims[:-1] + torch.Size([self.shape()]))

        input = input.flatten(start_dim=0, end_dim=-2)

        input = self.linear(input)
        input = self.norm_layer(input) if self.norm_layer is not None else input
        input = self.act(input) if self.act is not None else input

        input = input.view(tuple(out_dims))

        return input

    def shape(self, in_shape: int | None = None):
        L_out = self.cfg["out_dims"]

        return L_out
