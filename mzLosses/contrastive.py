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


import torch
import torch.nn as nn


class SoftNearestNeighbourLoss(nn.Module):

    def __init__(self, temp: float = 0.5, eps: float = 1e-10) -> None:
        super().__init__()
        self.temp = temp
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.flatten(start_dim=1), y.flatten(start_dim=0)

        b, features_dim = x.size()

        x = x/torch.norm(x, dim=1, keepdim=True) # Normalize the features

        # Create positives map
        pos_map = (~torch.eye(b, dtype=torch.bool)) * (y.unsqueeze(dim=0) == y.unsqueeze(dim=1))

        loss = torch.Tensor([0.]).float()

        tmp = torch.arange(b, dtype=torch.int32)
        for i in range(b):
            instance_feature = x[i, :].view(features_dim, 1) # [1, features_dim]
            oth_features = x[tmp != i, :] # [b-1, features_dim]
            pos_features = x[pos_map[i, :], :] # [pos_len, features_dim]

            pos_sim = -pos_features.matmul(instance_feature).squeeze(dim=-1) # [pos_len]
            oth_sim = -oth_features.matmul(instance_feature).squeeze(dim=-1) # [b-1]

            loss += ( (torch.sum(pos_sim.exp()/self.temp) + self.eps) / (torch.sum(oth_sim.exp()/self.temp)+self.eps) ).log()

        loss /= -b

        return loss
