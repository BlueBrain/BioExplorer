#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer
# <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

from bioexplorer import BioExplorer, Mesh, Vector3, Quaternion


def test_mesh():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'
    obj_folder = resource_folder + 'obj/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    mesh_source = obj_folder + 'capsule.obj'
    scale = Vector3(50, 50, 50)

    # Membrane
    protein_source = pdb_folder + 'membrane/popc.pdb'
    mesh = Mesh(mesh_source=mesh_source, protein_source=protein_source,
                density=5.0, surface_variable_offset=2.0)
    be.add_mesh('Mesh', mesh, scale=scale)

    # Receptors
    protein_source = pdb_folder + '6m1d.pdb'
    mesh = Mesh(
        mesh_source=mesh_source, protein_source=protein_source,
        density=0.02, surface_fixed_offset=5.0)
    be.add_mesh('Receptors', mesh, scale=scale)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
