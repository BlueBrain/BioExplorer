#!/usr/bin/env python
"""Test glucose"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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

from bioexplorer import BioExplorer, Volume, Protein, Vector3
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_glucose():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.set_general_settings(model_visibility_on_creation=False)
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    glucose_path = os.path.join(pdb_folder, 'glucose.pdb')

    # Scene parameters
    scene_size = Vector3(800.0, 800.0, 800.0)

    # Glucose
    glucose = Protein(
        name=bio_explorer.NAME_GLUCOSE,
        source=glucose_path,
        load_non_polymer_chemicals=True,
        occurrences=120000
    )

    volume = Volume(
        name=bio_explorer.NAME_GLUCOSE,
        shape=bio_explorer.ASSEMBLY_SHAPE_CUBE,
        shape_params=scene_size,
        protein=glucose
    )

    bio_explorer.add_volume(
        volume=volume,
        representation=bio_explorer.REPRESENTATION_ATOMS
    )

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
