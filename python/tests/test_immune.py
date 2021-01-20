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

def test_immune():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/immune/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)

    ''' Proteins '''
    glycan_radius_multiplier = 1.0
    glycan_add_sticks = True

    lactoferrin_path = pdb_folder + '1b0l.pdb'
    defensin_path = pdb_folder + '1ijv.pdb'

    ''' Scene parameters '''
    scene_size = 800

    ''' Lactoferrins'''
    lactoferrins = Protein(
        sources=[lactoferrin_path],
        load_non_polymer_chemicals=True,
        number_of_instances=150
    )

    lactoferrins_volume = Volume(
        name=be.NAME_LACTOFERRIN,
        size=Vector2(scene_size, scene_size),
        protein=lactoferrins
    )

    be.add_volume(
        volume=lactoferrins_volume,
        representation=be.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0)
    )

    ''' Defensins '''
    defensins = Protein(
        sources=[defensin_path],
        load_non_polymer_chemicals=True,
        number_of_instances=300
    )

    defensins_volume = Volume(
        name=be.NAME_DEFENSIN,
        size=Vector2(scene_size, scene_size),
        protein=defensins
    )

    be.add_volume(
        volume=defensins_volume,
        representation=be.REPRESENTATION_ATOMS,
        position=Vector3(0.0, scene_size / 2.0 - 200.0, 0.0),
        random_seed=3
    )

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)