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

from bioexplorer import BioExplorer, Surfactant, MolecularSystemAnimationParams, Vector3
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_surfactant():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    surfactant_folder = os.path.join(pdb_folder, 'surfactant')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)
    bio_explorer.core_api().set_camera(
        orientation=[0.0, 0.0, 0.0, 1.0], position=[0, 0, 200], target=[0, 0, 0])

    # Proteins
    protein_representation = bio_explorer.protein_representation.ATOMS

    head_source = os.path.join(surfactant_folder, '1pw9.pdb')
    branch_source = os.path.join(surfactant_folder, '1k6f.pdb')

    # SP-D
    surfactant_d = Surfactant(
        name='SP-D', surfactant_protein=bio_explorer.surfactant_type.PROTEIN_D,
        head_source=head_source, branch_source=branch_source)
    bio_explorer.add_surfactant(
        surfactant=surfactant_d, representation=protein_representation,
        position=Vector3(-50, 0, 0),
        animation_params=MolecularSystemAnimationParams(10)
    )

    # SP-A
    surfactant_a = Surfactant(
        name='SP-A', surfactant_protein=bio_explorer.surfactant_type.PROTEIN_A,
        head_source=head_source, branch_source=branch_source)
    bio_explorer.add_surfactant(
        surfactant=surfactant_a, representation=protein_representation,
        position=Vector3(50, 0, 0))

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
