#!/usr/bin/env python
"""Test surfactants"""

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

from bioexplorer import BioExplorer, Surfactant, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_surfactant():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/surfactant/'

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)
    bio_explorer.core_api().set_camera(
        orientation=[0.0, 0.0, 0.0, 1.0], position=[0, 0, 200], target=[0, 0, 0])

    # Proteins
    protein_representation = bio_explorer.REPRESENTATION_ATOMS

    head_source = pdb_folder + '1pw9.pdb'
    branch_source = pdb_folder + '1k6f.pdb'

    # SP-D
    surfactant_d = Surfactant(
        name='SP-D', surfactant_protein=bio_explorer.SURFACTANT_PROTEIN_D, head_source=head_source,
        branch_source=branch_source)
    bio_explorer.add_surfactant(
        surfactant=surfactant_d, representation=protein_representation, position=Vector3(-50, 0, 0),
        random_seed=10)

    # SP-A
    surfactant_a = Surfactant(
        name='SP-A', surfactant_protein=bio_explorer.SURFACTANT_PROTEIN_A, head_source=head_source,
        branch_source=branch_source)
    bio_explorer.add_surfactant(
        surfactant=surfactant_a, representation=protein_representation, position=Vector3(50, 0, 0))

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
