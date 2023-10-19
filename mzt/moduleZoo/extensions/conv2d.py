from typing import List, Tuple
import numpy as np
import torch
from ..convolution import ConvNormActivation2d, ConvInvertedBlock2d
from ..resblocks import ConvBottleNeckResidualBlock2d, ConvInvertedResidualBlock2d
from ..attention import SelfAttention2d, MultiHeadAttention2d, MultiHeadSelfAttention2d


def strip_to_numpy_(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    return x.detach().cpu().numpy()


def strip_to_numpy(
    x: np.ndarray | torch.Tensor | List[np.ndarray | torch.Tensor],
) -> np.ndarray:
    if isinstance(x, list):
        return [strip_to_numpy_(x_) for x_ in x]  # type: ignore

    return strip_to_numpy_(x)


_supported_objs = (
    ConvNormActivation2d
    | ConvInvertedBlock2d
    | ConvBottleNeckResidualBlock2d
    | ConvInvertedResidualBlock2d
    | SelfAttention2d
    | MultiHeadAttention2d
    | MultiHeadSelfAttention2d
)


def gt_rescale_conv_2d_(
    inst: _supported_objs,
    coords: np.ndarray | torch.Tensor,
    span: np.ndarray | torch.Tensor,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    # coords: [n, 2] : coordinates in the image frame, centered at top-left corner : y  x 
    # span: [2, 1] or [2, ] : [h, w]

    # Algorithm:
    #   Taken into account : All Factors

    coords, span = strip_to_numpy([coords, span])
    coords = coords.reshape((-1, 2))  # [n, 2]
    span = span.reshape((2,))  # [2, ]

    # TODO@ShivamPR21: Provide support for y_shape arguments in attention module
    out_span = np.array(list(inst.shape(tuple(span))), dtype=np.float32)  # type: ignore

    scaling = (out_span / span).reshape((1, 2))

    coords = np.minimum(
        coords * scaling, out_span
    )  # Minimum operation wraps output coordinates to the target bounds i.e. out_span

    return coords, out_span


def gt_rescale_conv_2d(
    inst: _supported_objs | List[_supported_objs],
    coords: np.ndarray | torch.Tensor,
    span: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    # coords: [n, 2] : coordinates in the image frame, centered at top-left corner : y  x 
    # span: [2, 1] or [2, ] : [h, w]

    # Algorithm:
    #   Taken into account : All Factors
    if not isinstance(inst, list):
        coords, _ = gt_rescale_conv_2d_(inst, coords, span)
        return coords

    for inst_ in inst:
        coords, span = gt_rescale_conv_2d_(inst_, coords, span)

    return coords
