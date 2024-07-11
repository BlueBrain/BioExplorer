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
import math
import random
import numpy as np

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value

def generate_random_point_on_torus(major_radius, minor_radius):
    phi = 2 * math.pi * random.random()  # Angle in the xy-plane
    theta = 2 * math.pi * random.random()  # Angle in the toroidal direction

    x = (major_radius + minor_radius * math.cos(theta)) * math.cos(phi)
    y = (major_radius + minor_radius * math.cos(theta)) * math.sin(phi)
    z = minor_radius * math.sin(theta)

    return (x, y, z)

def test_vector_fields():
    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    origins = list()
    targets = list()
    radii = list()
    for i in range(1500):

        x, y, z = generate_random_point_on_torus(10.0, 3.5)
        origin = Vector3(x, y ,z)

        cp = np.cross([x, y, z], [0, 0, 1])
        length = random.random() * 0.2
        target = Vector3(
            origin.x + cp[0] * length,
            origin.y + cp[1] * length, 
            origin.z + cp[2] * length
        )
        origins.append(origin)
        targets.append(target)
        radii.append(0.05)

    bio_explorer.core_api().set_renderer(
        current='basic',
        samples_per_pixel=1, subsampling=1, max_accum_frames=64)

    bio_explorer.add_cones(
        name='test',
        origins=origins, origins_radii=radii,
        targets=targets, targets_radii=radii,
        color=Vector3(1.0, 1.0, 1.0), opacity=1.0)
    
    bio_explorer.build_fields(
        voxel_size=0.5, density=1.0,
        data_type=bio_explorer.FIELD_DATA_TYPE_VECTOR
    )

    bio_explorer.core_api().set_renderer(
        current='vector_fields',
        samples_per_pixel=1, subsampling=1, max_accum_frames=16)
    params = bio_explorer.core_api().VectorFieldsRendererParams()
    print(params)
    params.cutoff = 150.0
    params.nb_ray_steps = 64
    params.nb_ray_refinement_steps = 16
    params.alpha_correction = 0.5
    bio_explorer.core_api().set_renderer_params(params)

    # Restore image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
