#!/usr/bin/env python
"""Test protein E"""

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

from bioexplorer import BioExplorer, Protein, Sugar, Quaternion
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_layout():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    glycan_folder = os.path.join(pdb_folder, 'glycans')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Resources

    protein_representation = be.protein_representation.ATOMS_AND_STICKS
    protein_radius_multiplier = 1.0
    glycan_representation = be.protein_representation.ATOMS_AND_STICKS
    glycan_radius_multiplier = 1.0

    # M Protein
    source = os.path.join(pdb_folder, 'QHD43418a.pdb')

    name = bio_explorer.NAME_PROTEIN_M
    m_protein = Protein(
        name=name,
        source=source,
        load_hydrogen=False,
        load_non_polymer_chemicals=False,
    )

    bio_explorer.add_protein(
        protein=m_protein,
        atom_radius_multiplier=protein_radius_multiplier,
        representation=protein_representation
    )

    # Glycans
    complex_folder = os.path.join(glycan_folder, 'complex')
    complex_paths = [
        os.path.join(complex_folder, '33.pdb'),
        os.path.join(complex_folder, '34.pdb'),
        os.path.join(complex_folder, '35.pdb'),
        os.path.join(complex_folder, '36.pdb')
    ]

    # High-mannose glycans on Protein M
    indices = [48, 66]
    complex_glycans = Sugar(
        rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
        assembly_name=name, name=bio_explorer.NAME_GLYCAN_COMPLEX,
        protein_name=name, source=complex_paths[0],
        site_indices=indices,
        representation=glycan_representation,
        atom_radius_multiplier=glycan_radius_multiplier
    )
    bio_explorer.add_glycan(complex_glycans)

    # Materials
    bio_explorer.apply_default_color_scheme(shading_mode=bio_explorer.SHADING_MODE_BASIC)

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
