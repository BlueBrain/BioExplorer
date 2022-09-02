#!/usr/bin/env python
"""Test mesh"""

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

from bioexplorer import BioExplorer, Cell, Membrane, Protein, AnimationParams, Vector2, Vector3, \
    Quaternion
import glob
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_mesh():
    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.set_general_settings(model_visibility_on_creation=False)
    print('BioExplorer version ' + bio_explorer.version())

    name = 'Suzanne'

    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    obj_folder = os.path.join(resource_folder, 'obj')
    membrane_folder = os.path.join(pdb_folder, 'membrane')
    lipids_folder = os.path.join(membrane_folder, 'lipids')
    transporters_folder = os.path.join(pdb_folder, 'transporters')
    mesh_source = os.path.join(obj_folder, 'suzanne.obj')
    scale = Vector3(2.5, 2.5, 2.5)

    # ACE2 receptor definition
    ace2_receptor = Protein(
        name=bio_explorer.NAME_TRANS_MEMBRANE + '_ACE2',
        source=os.path.join(pdb_folder, '6m18.pdb'),
        transmembrane_params=Vector2(1.0, 2.0),
        rotation=Quaternion(0.0, 1.0, 0.0, 0.0),
        animation_params=AnimationParams(1), occurrences=20)

    # GLUT3 definition
    transporter = Protein(
        name=bio_explorer.NAME_TRANS_MEMBRANE + '_GLUT3',
        source=os.path.join(transporters_folder, '4zwc.pdb'),
        transmembrane_params=Vector2(1.0, 2.0),
        rotation=Quaternion(0.707, 0.707, 0.0, 0.0),
        animation_params=AnimationParams(2), chain_ids=[1], occurrences=30)

    # Membrane definition
    pdb_lipids = glob.glob(os.path.join(lipids_folder, '*.pdb'))[:8]

    membrane = Membrane(
        lipid_sources=pdb_lipids, lipid_density=1.0,
        load_non_polymer_chemicals=True, load_bonds=True,
        animation_params=AnimationParams(0, 1, 0.025, 2, 0.5)
    )

    clipping_planes = [
        [0.0, 1.0, 0.0, 20],
        [1.0, 0.0, 0.0, 10],
    ]

    # Cell definition
    cell = Cell(
        name=name,
        shape=bio_explorer.ASSEMBLY_SHAPE_MESH,
        shape_params=scale,
        shape_mesh_source=mesh_source,
        membrane=membrane,
        proteins=[ace2_receptor, transporter]
    )

    # Add cell to scene
    bio_explorer.add_cell(
        cell=cell,
        clipping_planes=clipping_planes
    )


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
