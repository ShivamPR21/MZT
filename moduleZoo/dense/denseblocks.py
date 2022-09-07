from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn


class LinearNormActivation(nn.Sequential):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bias: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """_summary_

        Args:
            in_dims (int): _description_
            out_dims (int): _description_
            bias (bool, optional): _description_. Defaults to True.
            norm_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
            activation_layer (Optional[Callable[..., nn.Module]], optional): _description_. Defaults to None.
        """

        self.cfg: Dict[str, Any] = {'in_dims': in_dims,
                                    'out_dims': out_dims,
                                    'bias': bias}

        layers:List[nn.Module] = None

        layers = [
            nn.Linear(
                in_dims,
                out_dims,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_dims))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)

    def shape(self, in_shape: Optional[int] = None):
        L_out = self.cfg["out_dims"]

        return L_out
