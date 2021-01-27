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

from bioexplorer import BioExplorer, Surfactant, Vector3


def test_surfactant():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/surfactant/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)
    be.core_api().set_camera(
        orientation=[0.0, 0.0, 0.0, 1.0], position=[0, 0, 200], target=[0, 0, 0])

    ''' Proteins '''
    protein_representation = be.REPRESENTATION_ATOMS

    head_source = pdb_folder + '1pw9.pdb'
    branch_source = pdb_folder + '1k6f.pdb'

    ''' SP-D '''
    surfactant_d = Surfactant(name='SP-D', surfactant_protein=be.SURFACTANT_PROTEIN_D, head_source=head_source,
                              branch_source=branch_source)
    be.add_surfactant(surfactant=surfactant_d, representation=protein_representation, position=Vector3(-50, 0, 0),
                      random_seed=10)

    ''' SP-A '''
    surfactant_a = Surfactant(name='SP-A', surfactant_protein=be.SURFACTANT_PROTEIN_A, head_source=head_source,
                              branch_source=branch_source)
    be.add_surfactant(surfactant=surfactant_a,
                      representation=protein_representation, position=Vector3(50, 0, 0))

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
