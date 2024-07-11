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

from bioexplorer import BioExplorer, Volume, Protein, MolecularSystemAnimationParams, Vector3
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value


def test_immune():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    immune_folder = os.path.join(pdb_folder, 'immune')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Proteins
    lactoferrin_path = os.path.join(immune_folder, '1b0l.pdb')
    defensin_path = os.path.join(immune_folder, '1ijv.pdb')

    # Scene parameters
    scene_size = Vector3(800.0, 800.0, 800.0)

    # Lactoferrins
    lactoferrin = Protein(
        name=bio_explorer.NAME_LACTOFERRIN,
        source=lactoferrin_path,
        load_non_polymer_chemicals=True,
        occurrences=150
    )

    lactoferrins_volume = Volume(
        name=bio_explorer.NAME_LACTOFERRIN,
        shape=bio_explorer.ASSEMBLY_SHAPE_CUBE,
        shape_params=scene_size,
        protein=lactoferrin
    )

    bio_explorer.add_volume(
        volume=lactoferrins_volume,
        representation=bio_explorer.REPRESENTATION_ATOMS
    )

    # Defensins
    defensin = Protein(
        name=bio_explorer.NAME_DEFENSIN,
        source=defensin_path,
        load_non_polymer_chemicals=True,
        occurrences=300,
        animation_params=MolecularSystemAnimationParams(3)
    )

    defensins_volume = Volume(
        name=bio_explorer.NAME_DEFENSIN,
        shape=bio_explorer.ASSEMBLY_SHAPE_CUBE,
        shape_params=scene_size,
        protein=defensin
    )

    bio_explorer.add_volume(
        volume=defensins_volume,
        representation=bio_explorer.REPRESENTATION_ATOMS
    )

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
