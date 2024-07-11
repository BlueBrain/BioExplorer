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
