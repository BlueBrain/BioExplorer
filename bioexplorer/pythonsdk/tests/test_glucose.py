#!/usr/bin/env python
"""Test glucose"""

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

from bioexplorer import BioExplorer, Volume, Protein, Vector2, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_glucose():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    glucose_path = pdb_folder + 'glucose.pdb'

    # Scene parameters
    scene_size = 800.0

    # Glucose
    protein = Protein(
        sources=[glucose_path],
        load_non_polymer_chemicals=True,
        occurences=120000
    )

    volume = Volume(
        name=bio_explorer.NAME_GLUCOSE, size=scene_size, protein=protein
    )

    bio_explorer.add_volume(
        volume=volume,
        representation=bio_explorer.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0)
    )

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
