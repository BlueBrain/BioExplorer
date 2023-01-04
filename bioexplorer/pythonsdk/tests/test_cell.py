#!/usr/bin/env python
"""Test cell"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2023 Blue BrainProject / EPFL
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

from bioexplorer import BioExplorer, Cell, Membrane, Protein, Vector2, Vector3, Quaternion
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_cell():
    name = 'Cell'
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    membrane_folder = os.path.join(pdb_folder, 'membrane')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.set_general_settings(model_visibility_on_creation=False)

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    protein_representation = bio_explorer.REPRESENTATION_ATOMS

    # Membrane parameters
    membrane_size = Vector3(800.0, 80.0, 800.0)
    membrane_nb_receptors = 20

    # ACE2 Receptor
    ace2_receptor = Protein(
        name=name + '_' + bio_explorer.NAME_RECEPTOR,
        source=os.path.join(pdb_folder, '6m1d.pdb'),
        occurrences=membrane_nb_receptors,
        transmembrane_params=Vector2(-6.0, 5.0))

    membrane = Membrane(lipid_sources=[os.path.join(membrane_folder, 'popc.pdb')])

    cell = Cell(
        name=name,
        shape=bio_explorer.ASSEMBLY_SHAPE_SINUSOID,
        shape_params=membrane_size,
        membrane=membrane,
        proteins=[ace2_receptor])

    bio_explorer.add_cell(
        cell=cell, position=Vector3(4.5, -186, 7.0), rotation=Quaternion(1, 0, 0, 0),
        representation=protein_representation)

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
