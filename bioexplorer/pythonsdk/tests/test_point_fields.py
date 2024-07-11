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

from bioexplorer import BioExplorer, Vector3

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value

def test_fields():
    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Create dataset
    positions = list()
    radii = list()
    for x in range(10):
        for y in range(10):
            for z in range(10):
                positions.append(Vector3(x, y, z))
                radii.append(0.1)
    bio_explorer.add_spheres(
        name='Test data',
        positions=positions, radii=radii,
        color=Vector3(1,1,1), opacity=1.0)

    bio_explorer.build_fields(
        voxel_size=0.1, density=1.0,
        data_type=bio_explorer.FieldDataType.POINT)

    # Virus
    bio_explorer.core_api().set_renderer(
        current='point_fields',
        samples_per_pixel=1, subsampling=8, max_accum_frames=8)
    params = bio_explorer.core_api().PointFieldsRendererParams()
    params.cutoff = 2000
    params.max_steps = 2048
    params.threshold = 0.05
    params.step = 2.0
    bio_explorer.core_api().set_renderer_params(params)

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
