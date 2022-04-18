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

from .invertedresidual import Conv2DInvertedResidual
from .residual import (
    Conv2DInvertedResidualBlock,
    Conv2DResidualBlock,
    ConvResidualBlock1d,
)
from .shuffleresidual import Conv2DShuffleInvertedResidual

__all__ = ('Conv2DInvertedResidual', 'Conv2DShuffleInvertedResidual',
           'Conv2DResidualBlock', 'Conv2DInvertedResidualBlock', 'ConvResidualBlock1d')
