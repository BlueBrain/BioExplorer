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
    Shape(const Vector4fs& clippingPlanes);
    ~Shape();

    virtual Transformation getTransformation(
        const uint64_t occurence, const uint64_t nbOccurences,
        const RandomizationDetails& randDetails, const float offset) const = 0;

    virtual Transformation getTransformation(
        const uint64_t occurence, const uint64_t nbOccurences,
        const RandomizationDetails& randDetails, const float offset,
        const float morphingStep) const = 0;

    virtual bool isInside(const Vector3f& point) const = 0;

    float getSurface() const { return _surface; }

    Boxf getBounds() const { return _bounds; }

    static Quaterniond weightedRandomRotation(const Quaterniond& q,
                                              const size_t seed,
                                              const size_t index,
                                              const float s);

    /**
     * @brief Return a random float between -0.5 and 0.5
     *
     * @return float A random float between -0.5 and 0.5
     */
    static float rnd1();

    /**
     * @brief Return a predefined random float between -0.5 and 0.5
     *
     * @param index Index of the random float in a predefined array
     * @return float A random float between -0.5 and 0.5
     */
    static float rnd2(const uint64_t index);

    /**
     * @brief Return a controlled random float between -0.5 and 0.5, currently a
     * sinusoidal function
     *
     * @param index Index of the random float in a sinusoidal function
     * @return float A random float between -0.5 and 0.5
     */
    static float rnd3(const uint64_t index);

protected:
    /**
     * @brief Generate a random quaternion
     *
     * @param seed Seed to apply to the randomness
     * @return Quaterniond Random quaternion
     */
    Quaterniond randomQuaternion(const size_t seed) const;

protected:
    Boxf _bounds;
    float _surface;
    Vector4fs _clippingPlanes;
};

/**
 * @brief Get a random position in a 2D square along the X and Z axis
 *
 * @param center Center of the delimited plane in the 3D scene
 * @param size Size of the side of square
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getPlanarPosition(const Vector3f& position, const float size,
                                 const RandomizationDetails& randInfo);

/**
 * @brief Get the Cubic Position object
 *
 * @param center Center of the cube in the 3D scene
 * @param size Size of the side of cube
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getCubicPosition(const Vector3f& center, const Vector3f& size,
                                const RandomizationDetails& randInfo);

/**
 * @brief Get the Cubic Position object
 *
 * @param x Position along the x axis
 * @param z Position along the z axis
 * @return float Position along the y axis
 */
float sinusoide(const float x, const float z);

/**
 * @brief Get the Sinosoidal Position object
 *
 * @param center Center of the sinosoidal function in the 3D scene
 * @param size Size of the side of sinosoidal function
 * @param amplitude Amplitude of the sinosoidal function
 * @param occurence Occurence of the position amongst the maximum of occurences
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getSinosoidalPosition(const Vector3f& center,
                                     const Vector2f& size,
                                     const float amplitude,
                                     const size_t occurence,
                                     const RandomizationDetails& randInfo);

/**
 * @brief Get the Fan Position object
 *
 * @param center Center of the fan in the 3D scene
 * @param radius Radius of the fan in the 3D scene
 * @param occurence Occurence of the position amongst the maximum of occurences
 * (see next parameters)
 * @param occurences Maximum number of occurences on the sphere
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getFanPosition(const Vector3f& center, const float radius,
                              const size_t occurence, const size_t occurences,
                              const RandomizationDetails& randInfo);

/**
 * @brief Get the Bezier Position object
 *
 * @param points Points defining the Bezier curve
 * @param scale Scale to apply to the points
 * @param t Value of t along the Bezier curve (0..1)
 * @return Transformation of the position and rotation on the Bezier curve
 */
Transformation getBezierPosition(const Vector3fs& points, const float scale,
                                 const float t);

/**
 * @brief Get the Spherical To Planar Position object
 *
 * @param center Center of the fan in the 3D scene
 * @param radius Radius of the fan in the 3D scene
 * @param occurence Occurence of the position amongst the maximum of occurences
 * (see next parameters)
 * @param occurences Maximum number of occurences on the sphere
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @param morphingStep
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getSphericalToPlanarPosition(
    const Vector3f& center, const float radius, const size_t occurence,
    const size_t occurences, const RandomizationDetails& randInfo,
    const float morphingStep);

} // namespace common
} // namespace bioexplorer
