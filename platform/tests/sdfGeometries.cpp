
/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <platform/core/common/geometry/SDFGeometry.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("bounding_box")
{
    const auto sphere = core::createSDFSphere({1.0f, 1.0f, 1.0f}, 1.0f);
    const auto conePill = core::createSDFConePill({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 2.0f, 1.0f);
    const auto pill = core::createSDFPill({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 2.0f);

    const auto boxSphere = getSDFBoundingBox(sphere);
    const auto boxConePill = getSDFBoundingBox(conePill);
    const auto boxPill = getSDFBoundingBox(pill);

    CHECK_EQ(boxSphere.getMin(), core::Vector3d(0.0, 0.0, 0.0));
    CHECK_EQ(boxSphere.getMax(), core::Vector3d(2.0, 2.0, 2.0));

    CHECK_EQ(boxConePill.getMin(), core::Vector3d(-2.0, -2.0, -2.0));
    CHECK_EQ(boxConePill.getMax(), core::Vector3d(2.0, 2.0f, 2.0));

    CHECK_EQ(boxPill.getMin(), core::Vector3d(-2.0, -2.0, -2.0));
    CHECK_EQ(boxPill.getMax(), core::Vector3d(3.0, 3.0, 3.0));
}
