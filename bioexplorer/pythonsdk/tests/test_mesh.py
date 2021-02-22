#!/usr/bin/env python
"""Test mesh"""

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

from bioexplorer import BioExplorer, Mesh, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_mesh():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'
    obj_folder = resource_folder + 'obj/'

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset()
    print('BioExplorer version ' + bio_explorer.version())

    mesh_source = obj_folder + 'suzanne.obj'
    scale = Vector3(5, 5, 5)

    # Membrane
    protein_source = pdb_folder + 'membrane/popc.pdb'
    mesh = Mesh(mesh_source=mesh_source, protein_source=protein_source,
                density=5.0, surface_variable_offset=2.0)
    bio_explorer.add_mesh('Mesh', mesh, scale=scale)

    # Receptors
    protein_source = pdb_folder + '6m1d.pdb'
    mesh = Mesh(
        mesh_source=mesh_source, protein_source=protein_source,
        density=0.02, surface_fixed_offset=5.0)
    bio_explorer.add_mesh('Receptors', mesh, scale=scale)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
