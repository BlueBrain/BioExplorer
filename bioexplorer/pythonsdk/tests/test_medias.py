# !/usr/bin/env python
"""MediaMaker test"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2021 Blue BrainProject / EPFL
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

from bioexplorer import BioExplorer, MovieMaker


def test_movie_maker():
    bio_explorer = BioExplorer('localhost:5000')
    core = bio_explorer.core_api()
    movie_maker = MovieMaker(bio_explorer)
    movie_maker.version()

    core.set_camera(current='bio_explorer_perspective')

    control_points = [
        {
            'apertureRadius': 4.224657298716223e-290,
            'direction': [0.0, 0.0, -1.0],
            'focusDistance': 0.0,
            'origin': [0.5, 0.5, 1.5],
            'up': [0.0, 1.0, 0.0]
        },
        {
            'apertureRadius': 0.0,
            'direction': [-0.4823790279394327, -0.35103051457124496, -0.8025509648888691],
            'focusDistance': 0.0,
            'origin': [2.020997896788385, 1.606840561979088, 3.0305377285488593],
            'up': [-0.1993924108090585, 0.9361435664152715, -0.2896167978053891]
        }
    ]

    movie_maker.build_camera_path(
        control_points=control_points, nb_steps_between_control_points=50,
        smoothing_size=50)
    print(movie_maker.get_nb_frames())

    movie_maker.set_current_frame(10)
    movie_maker.create_movie(
        path='/tmp', size=[512, 512], samples_per_pixel=16, start_frame=10, end_frame=20)

    movie_maker.set_current_frame(20)
    movie_maker.create_snapshot(
        renderer='bio_explorer',
        path='/tmp', base_name='test_20', size=[512, 512], samples_per_pixel=16)

    movie_maker.set_current_frame(30)
    movie_maker.create_snapshot(
        renderer='bio_explorer',
        path='/tmp', base_name='test_30', size=[512, 512], samples_per_pixel=16)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
