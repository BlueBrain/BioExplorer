#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, Cyrille Favreau <cyrille.favreau@epfl.ch>
#
# This file is part of BioExplorer
# <https://github.com/BlueBrain/BioExplorer>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# All rights reserved. Do not distribute without further notice.

from bioexplorer import BioExplorer, MovieMaker

be = BioExplorer('localhost:5000')
print('BioExplorer version ' + be.version())

# Create scene
be.add_coronavirus(name='Coronavirus', resource_folder='test_files/')

# Export frames
control_points = [{'apertureRadius': 0.0, 'focusDistance': 1e6,
                   'origin': [-6.017819822266641, -2.344840659134118, 209.77415769873744],
                   'direction': [0.0, 0.0, -1.0], 'up': [0.0, 1.0, 0.0]},
                  {'apertureRadius': 0.0, 'focusDistance': 1e6,
                   'origin': [78.24581560831695, 60.38818629612713, 102.90041257728218],
                   'direction': [-0.551634920162921, -0.42315843749891463, -0.718773852912253],
                   'up': [-0.1434787205331168, 0.8970466735838447, -0.4179965576010632]}]
mm = MovieMaker(be)
mm.build_camera_path(control_points=control_points, nb_steps_between_control_points=10, smoothing_size=10)
mm.export_frames(path='/tmp', size=[256,256], samples_per_pixel=16)
