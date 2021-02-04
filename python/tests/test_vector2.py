#!/usr/bin/env python
"""Test Vector3"""

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

from bioexplorer import Vector2


def test_vector2_1():
    try:
        Vector2(1)
        assert False
    except RuntimeError:
        assert True


def test_vector2_2():
    try:
        Vector2(1, 2)
        assert True
    except RuntimeError:
        assert False


def test_vector2_3():
    try:
        Vector2(1, 2, 3)
        assert False
    except RuntimeError:
        assert True


def test_vector2_to_list():
    try:
        vector = Vector2(0, 1)
        assert vector.to_list() == [0, 1]
    except RuntimeError:
        assert True


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
