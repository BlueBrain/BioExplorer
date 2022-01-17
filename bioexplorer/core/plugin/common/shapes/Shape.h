/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#pragma once

#include <plugin/common/Types.h>

#include <brayns/common/Transformation.h>

namespace bioexplorer
{
namespace common
{
using namespace details;
using namespace brayns;

class Shape
{
public:
    Shape(const Vector4ds& clippingPlanes);
    ~Shape();

    virtual Transformation getTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const AnimationDetails& animationDetails,
        const double offset = 0.0) const = 0;

    virtual bool isInside(const Vector3d& point) const = 0;

    double getSurface() const { return _surface; }

    Boxf getBounds() const { return _bounds; }

    static Quaterniond weightedRandomRotation(const Quaterniond& q,
                                              const uint64_t seed,
                                              const uint64_t index,
                                              const double s);

    /**
     * @brief Return a random double between -0.5 and 0.5
     *
     * @return double A random double between -0.5 and 0.5
     */
    static double rnd1();

    /**
     * @brief Return a predefined random double between -0.5 and 0.5
     *
     * @param index Index of the random double in a predefined array
     * @return double A random double between -0.5 and 0.5
     */
    static double rnd2(const uint64_t index);

    /**
     * @brief Return a controlled random double between -0.5 and 0.5, currently
     * a sinusoidal function
     *
     * @param index Index of the random double in a sinusoidal function
     * @return double A random double between -0.5 and 0.5
     */
    static double rnd3(const uint64_t index);

protected:
    /**
     * @brief Generate a random quaternion
     *
     * @param seed Seed to apply to the randomness
     * @return Quaterniond Random quaternion
     */
    Quaterniond randomQuaternion(const uint64_t seed) const;

protected:
    Boxf _bounds;
    double _surface;
    Vector4ds _clippingPlanes;
};
} // namespace common
} // namespace bioexplorer
