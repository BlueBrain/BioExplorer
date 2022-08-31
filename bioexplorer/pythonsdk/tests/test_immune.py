#!/usr/bin/env python
"""Test immune system"""

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

from bioexplorer import BioExplorer, Volume, Protein, AnimationParams, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_immune():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/immune/'

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.set_general_settings(model_visibility_on_creation=False)
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    lactoferrin_path = pdb_folder + '1b0l.pdb'
    defensin_path = pdb_folder + '1ijv.pdb'

    # Scene parameters
    scene_size = Vector3(800.0, 800.0, 800.0)

    # Lactoferrins
    lactoferrin = Protein(
        name=bio_explorer.NAME_LACTOFERRIN,
        source=lactoferrin_path,
        load_non_polymer_chemicals=True,
        occurrences=150
    )

    lactoferrins_volume = Volume(
        name=bio_explorer.NAME_LACTOFERRIN,
        shape=bio_explorer.ASSEMBLY_SHAPE_CUBE,
        shape_params=scene_size,
        protein=lactoferrin
    )

    bio_explorer.add_volume(
        volume=lactoferrins_volume,
        representation=bio_explorer.REPRESENTATION_ATOMS
    )

    # Defensins
    defensin = Protein(
        name=bio_explorer.NAME_DEFENSIN,
        source=defensin_path,
        load_non_polymer_chemicals=True,
        occurrences=300,
        animation_params=AnimationParams(3)
    )

    defensins_volume = Volume(
        name=bio_explorer.NAME_DEFENSIN,
        shape=bio_explorer.ASSEMBLY_SHAPE_CUBE,
        shape_params=scene_size,
        protein=defensin
    )

    bio_explorer.add_volume(
        volume=defensins_volume,
        representation=bio_explorer.REPRESENTATION_ATOMS
    )

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
