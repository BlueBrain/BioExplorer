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

from bioexplorer import BioExplorer, Protein, Surfactant, Vector3, Quaternion
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_layout():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')
    immune_folder = os.path.join(pdb_folder, 'immune')
    surfactant_folder = os.path.join(pdb_folder, 'surfactant')

    bio_explorer = BioExplorer('localhost:5000')
    core = bio_explorer.core_api()
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    line_surfactant = 5
    line_virus = 25
    line_defense = 45

    # Camera
    core.set_camera(
        current='orthographic',
        orientation=[0.0, 0.0, 0.0, 1.0],
        position=[23.927943790322814, 24.84577580212592, 260.43975983632527],
        target=[23.927943790322814, 24.84577580212592, 39.93749999999999]
    )
    params = core.OrthographicCameraParams()
    params.height = 55
    core.set_camera_params(params)

    # Grid
    bio_explorer.add_grid(
        min_value=0, max_value=100, interval=1, radius=0.005, colored=False,
        position=Vector3(-10.0, -10.0, -10.0)
    )

    # Layout
    virus_protein_s_open = Protein(
        name='Protein S (open)',
        source=os.path.join(pdb_folder, '6vyb.pdb'),
        rotation=Quaternion(0.707, 0.707, 0.0, 0.0)
    )
    bio_explorer.add_protein(
        protein=virus_protein_s_open,
        position=Vector3(5.0, line_virus, 0.0)
    )

    virus_protein_s_closed = Protein(
        name='Protein S (closed)',
        source=os.path.join(pdb_folder, '6vyb.pdb'),
        rotation=Quaternion(0.707, 0.707, 0.0, 0.0)
    )
    bio_explorer.add_protein(
        protein=virus_protein_s_closed,
        position=Vector3(20.0, line_virus, 0.0)
    )

    # Protein M (QHD43419)
    virus_protein_m = Protein(
        name='Protein M',
        source=os.path.join(pdb_folder, 'QHD43419a.pdb')
    )
    bio_explorer.add_protein(
        protein=virus_protein_m,
        position=Vector3(35.0, line_virus, 0.0)
    )

    # Protein E (QHD43418 P0DTC4)
    virus_protein_e = Protein(
        name='Protein E',
        source=os.path.join(pdb_folder, 'QHD43418a.pdb')
    )
    bio_explorer.add_protein(
        protein=virus_protein_e,
        position=Vector3(45.0, line_virus, 0.0)
    )

    # Lactoferrin
    lactoferrin = Protein(
        name='Lactoferrin',
        source=os.path.join(immune_folder, '1b0l.pdb')
    )
    bio_explorer.add_protein(
        protein=lactoferrin,
        position=Vector3(5.0, line_defense, 0.0)
    )

    # Defensin
    defensin = Protein(
        name='Defensin',
        source=os.path.join(immune_folder, '1ijv.pdb')
    )
    bio_explorer.add_protein(
        protein=defensin,
        position=Vector3(20.0, line_defense, 0.0)
    )

    # Glucose
    glucose = Protein(
        name='Glucose',
        source=os.path.join(pdb_folder, 'glucose.pdb'),
        load_non_polymer_chemicals=True,
        rotation=Quaternion(0.0, 0.0, 0.707, 0.707)
    )
    bio_explorer.add_protein(
        protein=glucose,
        position=Vector3(30.0, line_defense, 0.0)
    )

    # ACE2 Receptor
    ace2_receptor = Protein(
        name='ACE2 receptor',
        source=os.path.join(pdb_folder, '6m18.pdb'),
        rotation=Quaternion(0.0, 0.0, 0.707, -0.707)
    )
    bio_explorer.add_protein(
        protein=ace2_receptor,
        position=Vector3(45.0, line_defense - 2.5, 0.0)
    )

    # Surfactant
    head_source = os.path.join(surfactant_folder, '1pw9.pdb')
    branch_source = os.path.join(surfactant_folder, '1k6f.pdb')
    surfactant_d = Surfactant(
        name='Surfactant',
        surfactant_protein=bio_explorer.surfactant_type.PROTEIN_A,
        head_source=head_source,
        branch_source=branch_source
    )

    bio_explorer.add_surfactant(
        surfactant=surfactant_d,
        position=Vector3(50.0, line_surfactant, 0.0)
    )

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
