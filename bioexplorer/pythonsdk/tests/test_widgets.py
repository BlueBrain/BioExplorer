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
    widgets.environment_maps('')


def test_widgets_advanced_camera_settings():
    widgets.advanced_camera_settings(is_threaded=False)


def test_widgets_display_rendering_settings():
    widgets.rendering_settings()


def test_widgets_advanced_rendering_settings():
    widgets.advanced_rendering_settings(is_threaded=False)


def test_widgets_focal_distance():
    widgets.focal_distance()


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
