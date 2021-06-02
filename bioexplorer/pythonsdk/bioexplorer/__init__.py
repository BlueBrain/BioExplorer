#!/usr/bin/env python
"""Initializer"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2021 Blue BrainProject / EPFL
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

from .bio_explorer import BioExplorer, Volume, MeshBasedMembrane, AssemblyProtein, Protein, \
    Sugars, RNASequence, Cell, ParametricMembrane, Surfactant, Virus, Vector2, Vector3, \
    Quaternion
from .movie_maker import MovieMaker
from .transfer_function import TransferFunction
from .notebook_widgets import Widgets
from .version import VERSION as __version__

__all__ = [
    'Widgets', 'BioExplorer', 'MeshBasedMembrane', 'Protein', 'AssemblyProtein', 'Sugars',
    'RNASequence', 'ParametricMembrane', 'Volume', 'Surfactant', 'Cell', 'Vector2', 'Vector3',
    'Quaternion', 'Virus', 'MovieMaker', 'TransferFunction', '__version__']
