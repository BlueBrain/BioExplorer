/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// NOTE: Most of this code was generated with codeGPT

#define BOOST_TEST_MODULE TestIsClipped
#include <bioexplorer/core/plugin/common/Utils.h>
#include <boost/test/unit_test.hpp>

using namespace brayns;
using namespace bioexplorer;
using namespace common;

BOOST_AUTO_TEST_SUITE(TestIsClippedSuite)

// Test case that checks if a point outside all clipping planes is not clipped
BOOST_AUTO_TEST_CASE(TestIsClippedOutsidePlanes)
{
    Vector3d position{10.0, 20.0, 30.0};
    Vector4ds clippingPlanes{{-1.0, 0.0, 0.0, -2.0},
                             {0.0, -1.0, 0.0, -3.0},
                             {0.0, 0.0, -1.0, -4.0}};
    BOOST_CHECK(!isClipped(position, clippingPlanes));
}

// Test case that checks if a point inside all clipping planes is clipped
BOOST_AUTO_TEST_CASE(TestIsClippedInsidePlanes)
{
    Vector3d position{4.0, 5.0, 6.0};
    Vector4ds clippingPlanes{{1.0, 0.0, 0.0, -5.0},
                             {0.0, 1.0, 0.0, -6.0},
                             {0.0, 0.0, 1.0, -7.0}};
    BOOST_CHECK(isClipped(position, clippingPlanes));
}

// Test case that checks if a point outside some clipping planes is clipped
BOOST_AUTO_TEST_CASE(TestIsClippedOutsideSomePlanes)
{
    Vector3d position{4.0, 5.0, 6.0};
    Vector4ds clippingPlanes{{1.0, 0.0, 0.0, -5.0},
                             {0.0, -1.0, 0.0, -6.0},
                             {0.0, 0.0, 1.0, -7.0}};
    BOOST_CHECK(isClipped(position, clippingPlanes));
}

// Test case that checks if a point inside some clipping planes is not clipped
BOOST_AUTO_TEST_CASE(TestIsClippedInsideSomePlanes)
{
    Vector3d position{0.0, 0.0, 0.0};
    Vector4ds clippingPlanes{{1.0, 0.0, 0.0, 1.0},
                             {0.0, -1.0, 0.0, 1.0},
                             {0.0, 0.0, 1.0, 1.0}};
    BOOST_CHECK(!isClipped(position, clippingPlanes));
}

BOOST_AUTO_TEST_SUITE_END()
