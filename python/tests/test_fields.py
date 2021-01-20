#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, EPFL/Blue Brain Project
# All rights reserved. Do not distribute without permission.
# Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

from bioexplorer import BioExplorer


def test_fields():
    import os
    resource_folder = os.getcwd() + 'tests/test_files/'
    fields_folder = resource_folder + 'fields/'

    be = BioExplorer('localhost:5000')
    be.reset()
    print('BioExplorer version ' + be.version())

    ''' Suspend image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=0)

    ''' Import from file '''
    be.import_fields_from_file(fields_folder + 'protein.fields')

    ''' Virus '''
    be.core_api().set_renderer(current='bio_explorer_fields', samples_per_pixel=1, subsampling=8, max_accum_frames=8)
    params = be.core_api().BioExplorerFieldsRendererParams()
    params.cutoff = 2000
    params.max_steps = 2048
    params.threshold = 0.05
    params.step = 2.0
    be.core_api().set_renderer_params(params)

    ''' Restore image streaming '''
    be.core_api().set_application_parameters(image_stream_fps=20)


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
