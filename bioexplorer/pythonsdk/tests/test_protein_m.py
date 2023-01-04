#!/usr/bin/env python
"""Test protein M"""

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
    bio_explorer.set_general_settings(model_visibility_on_creation=False)
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Resources

    protein_representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS
    protein_radius_multiplier = 1.0
    glycan_representation = BioExplorer.REPRESENTATION_ATOMS_AND_STICKS
    glycan_radius_multiplier = 1.0

    # M Protein
    source = os.path.join(pdb_folder, 'QHD43419a.pdb')

    name = bio_explorer.NAME_PROTEIN_M
    m_protein = Protein(
        name=name,
        source=source,
        load_hydrogen=False,
        load_non_polymer_chemicals=False
    )

    bio_explorer.add_protein(
        protein=m_protein,
        atom_radius_multiplier=protein_radius_multiplier,
        representation=protein_representation
    )

    # Glycans
    high_mannose_folder = os.path.join(glycan_folder, 'high-mannose')
    high_mannose_paths = [
        os.path.join(high_mannose_folder, '1.pdb'),
        os.path.join(high_mannose_folder, '2.pdb'),
        os.path.join(high_mannose_folder, '3.pdb'),
        os.path.join(high_mannose_folder, '4.pdb')
    ]

    # High-mannose glycans on Protein M
    indices = [5]
    high_mannose_glycans = Sugar(
        rotation=Quaternion(0.707, 0.0, 0.0, 0.707),
        assembly_name=name, name=bio_explorer.NAME_GLYCAN_HIGH_MANNOSE,
        protein_name=name, source=high_mannose_paths[0],
        site_indices=indices,
        representation=glycan_representation,
        atom_radius_multiplier=glycan_radius_multiplier
    )
    bio_explorer.add_glycan(high_mannose_glycans)

    # Materials
    bio_explorer.apply_default_color_scheme(shading_mode=bio_explorer.SHADING_MODE_BASIC)

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
