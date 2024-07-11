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

from bioexplorer import BioExplorer, Cell, Membrane, Protein, MolecularSystemAnimationParams, Vector2, Vector3, \
    Quaternion
import glob
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_mesh():
    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
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
        animation_params=MolecularSystemAnimationParams(1), occurrences=20)

    # GLUT3 definition
    transporter = Protein(
        name=bio_explorer.NAME_TRANS_MEMBRANE + '_GLUT3',
        source=os.path.join(transporters_folder, '4zwc.pdb'),
        transmembrane_params=Vector2(1.0, 2.0),
        rotation=Quaternion(0.707, 0.707, 0.0, 0.0),
        animation_params=MolecularSystemAnimationParams(2), chain_ids=[1], occurrences=30)

    # Membrane definition
    pdb_lipids = glob.glob(os.path.join(lipids_folder, '*.pdb'))[:8]

    membrane = Membrane(
        lipid_sources=pdb_lipids, lipid_density=1.0,
        load_non_polymer_chemicals=True, load_bonds=True,
        animation_params=MolecularSystemAnimationParams(0, 1, 0.025, 2, 0.5)
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
