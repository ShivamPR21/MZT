'''
Copyright (C) 2021  Shivam Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from typing import List

import torch.nn as nn
from torch import Tensor

from ..moduleZoo.resblocks import ConvInvertedResidualBlock2d

#TODO@ShivamPR21: #5 Adapt model based on changed moduleZoo API
# class ShuffleNet(nn.Module):
#     def __init__(self,
#                  in_channel: int,
#                  settings: List[List[int]] = None
#                  ) -> None:
#         super().__init__()

#         if settings is None:
#             settings = [
#                 #k, s, r, exp
#                 [3, 2, 1, 1, 2],
#                 [3, 1, 3, 1, 2],
#                 [3, 2, 1, 1, 2],
#                 [3, 1, 7, 1, 2],
#                 [3, 2, 1, 1, 2],
#                 [3, 1, 3, 1, 2]]

#         features : List[nn.Module] = []
#         for k, s, r, e, g in settings:
#             for _ in range(r):
#                 features.append(
#                     ShuffleInvertedResidual(in_channel, in_channel, e,
#                                             g, k, s, nn.BatchNorm2d, nn.ReLU6)
#                 )
#                 if s == 2:
#                     in_channel *= 2

#         self.features = nn.Sequential(*features)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.features(x)
#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)
