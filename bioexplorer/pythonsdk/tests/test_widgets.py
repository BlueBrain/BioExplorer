#!/usr/bin/env python
"""Test Widgets"""

# -*- coding: utf-8 -*-

# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2022 Blue BrainProject / EPFL
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
