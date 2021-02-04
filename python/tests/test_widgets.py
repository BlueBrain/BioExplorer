#!/usr/bin/env python
"""Test Widgets"""

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


import time
from bioexplorer import BioExplorer, Widgets

bio_explorer = BioExplorer('localhost:5000')
widgets = Widgets(bio_explorer)


def test_widgets_environment_maps():
    widgets.display_environment_maps('')


def test_widgets_advanced_camera_settings():
    widgets.display_advanced_camera_settings(is_threaded=False)


def test_widgets_display_rendering_settings():
    widgets.display_rendering_settings()


def test_widgets_advanced_rendering_settings():
    widgets.display_advanced_rendering_settings(is_threaded=False)


def test_widgets_focal_distance():
    widgets.display_focal_distance()


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
