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

from bioexplorer import Quaternion


def test_quaternion_1():
    try:
        Quaternion(1)
        assert False
    except RuntimeError:
        assert True


def test_quaternion_2():
    try:
        Quaternion(1, 2)
        assert False
    except RuntimeError:
        assert True


def test_quaternion_3():
    try:
        Quaternion(1, 2, 3)
        assert False
    except RuntimeError:
        assert True


def test_quaternion_4():
    try:
        Quaternion(1, 2, 3, 4)
        assert True
    except RuntimeError:
        assert False


def test_quaternion_to_list():
    try:
        q = Quaternion(0, 1, 2, 3)
        assert q.to_list() == [0, 1, 2, 3]
    except RuntimeError:
        assert True


if __name__ == '__main__':
    import nose

    nose.run(defaultTest=__name__)
