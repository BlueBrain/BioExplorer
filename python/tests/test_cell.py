#!/usr/bin/env python
"""Test cell"""

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

from bioexplorer import BioExplorer, Cell, Membrane, Protein, Vector2, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_cell():
    resource_folder = 'tests/test_files/'
    pdb_folder = resource_folder + 'pdb/'

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset()

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    protein_representation = bio_explorer.REPRESENTATION_ATOMS

    # Membrane parameters
    membrane_size = 800
    membrane_height = 80
    membrane_nb_receptors = 20
    membrane_nb_lipids = 1200000

    # ACE2 Receptor
    ace2_receptor = Protein(
        sources=[pdb_folder + '6m1d.pdb'],
        occurences=membrane_nb_receptors,
        position=Vector3(0.0, 6.0, 0.0))

    membrane = Membrane(
        sources=[pdb_folder + 'membrane/popc.pdb'],
        occurences=membrane_nb_lipids)

    cell = Cell(
        name='Cell',
        size=Vector2(membrane_size, membrane_height),
        shape=bio_explorer.ASSEMBLY_SHAPE_SINUSOIDAL,
        membrane=membrane, receptor=ace2_receptor)

    bio_explorer.add_cell(
        cell=cell, position=Vector3(4.5, -186, 7.0),
        representation=protein_representation, random_seed=1)

    # Set rendering settings
    bio_explorer.core_api().set_renderer(
        background_color=[96 / 255, 125 / 255, 139 / 255], current='bio_explorer',
        samples_per_pixel=1, subsampling=4, max_accum_frames=64)
    params = bio_explorer.core_api().BioExplorerRendererParams()
    params.shadows = 0.75
    params.soft_shadows = 1.0
    bio_explorer.core_api().set_renderer_params(params)

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
