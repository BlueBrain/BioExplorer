#!/usr/bin/env python
"""Test magnetic fields"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2023 Blue BrainProject / EPFL
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

from bioexplorer import BioExplorer
import os

# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=dangerous-default-value

def test_fields():
    resource_folder = os.path.abspath('./tests/test_files')
    fields_folder = os.path.join(resource_folder, 'fields')

    bio_explorer = BioExplorer('localhost:5000')
    bio_explorer.reset_scene()
    bio_explorer.start_model_loading_transaction()
    print('BioExplorer version ' + bio_explorer.version())

    # Suspend image streaming
    bio_explorer.core_api().set_application_parameters(image_stream_fps=0)

    # Import from file
    bio_explorer.import_fields_from_file(os.path.join(fields_folder, 'receptor.fields'))

    # Virus
    bio_explorer.core_api().set_renderer(
        current='bio_explorer_fields',
        samples_per_pixel=1, subsampling=8, max_accum_frames=8)
    params = bio_explorer.core_api().BioExplorerFieldsRendererParams()
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
