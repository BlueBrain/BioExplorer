# Copyright 2020 - 2023 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
