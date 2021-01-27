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

from bioexplorer import BioExplorer, Volume, Protein, Vector2, Vector3


def test_glucose():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)

    ''' Proteins '''
    glycan_radius_multiplier = 1.0
    glycan_add_sticks = True

    glucose_path = pdb_folder + 'glucose.pdb'

    ''' Scene parameters '''
    scene_size = 800

    ''' Glucose'''
    protein = Protein(
        sources=[glucose_path],
        load_non_polymer_chemicals=True,
        occurences=120000
    )

    volume = Volume(
        name=be.NAME_GLUCOSE,
        size=Vector2(scene_size, scene_size),
        protein=protein
    )

    be.add_volume(
        volume=volume,
        representation=be.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0)
    )

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
