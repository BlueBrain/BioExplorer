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

from bioexplorer import BioExplorer, Protein, Quaternion
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring


def test_layout():
    resource_folder = os.path.abspath('./tests/test_files')
    pdb_folder = os.path.join(resource_folder, 'pdb')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Camera
    brayns = bio_explorer.core_api()
    brayns.set_camera(
        current='orthographic',
        orientation=[0.0, 0.0, 0.0, 1.0],
        position=[23.927943790322814, 24.84577580212592, 260.43975983632527],
        target=[23.927943790322814, 24.84577580212592, 39.93749999999999]
    )
    params = brayns.OrthographicCameraParams()
    params.height = 55
    brayns.set_camera_params(params)

    # ACE2 Receptor
    name = 'ACE2 receptor'
    ace2_receptor = Protein(
        name=name,
        source=os.path.join(pdb_folder, '6m1d.pdb'))
    bio_explorer.add_protein(
        ace2_receptor,
        rotation=Quaternion(0.5, 0.5, 1.0, 0.0))

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
