#!/usr/bin/env python
"""Test Vector3"""

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

from bioexplorer import Vector3


def test_vector3_1():
    try:
        Vector3(1)
        assert False
    except RuntimeError:
        assert True


def test_vector3_2():
    try:
        Vector3(1, 2)
        assert False
    except RuntimeError:
        assert True


def test_vector3_3():
    try:
        Vector3(1, 2, 3)
        assert True
    except RuntimeError:
        assert False


def test_vector3_4():
    try:
        Vector3(1, 2, 3, 4)
        assert False
    except RuntimeError:
        assert True


def test_vector3_to_list():
    try:
        vector = Vector3(0, 1, 2)
        assert vector.to_list() == [0, 1, 2]
    except RuntimeError:
        assert True


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
