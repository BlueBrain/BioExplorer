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

#include <brayns/common/types.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
using namespace brayns;

/**
 * @brief Left trim of a string
 * @param s String to trim
 * @return Trimmed string
 */
std::string& ltrim(std::string& s);

/**
 * @brief Right trim of a string
 * @param s String to trim
 * @return Trimmed string
 */
std::string& rtrim(std::string& s);

/**
 * @brief Left and right trim of a string
 * @param s String to trim
 * @return Trimmed string
 */
std::string& trim(std::string& s);

/**
 * @brief isClipped Determine if a 3d position is inside a volume defined by
 * clipping planes
 * @param position Position to check
 * @param clippingPlanes Clipping planes defining the volume
 * @return True if the position does not belong to the volume, false otherwise
 */
bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes);

/**
 * @brief getSphericalPosition Provide a random position and rotation on a
 * sphere
 * @param rnd Random seed for the position on the sphere
 * @param center Center of the sphere in the 3D scene
 * @param radius Radius of the sphere
 * @param occurence Occurence of the position amongst the maximum of occurences
 * (see next parameters)
 * @param occurences Maximum number of occurences on the sphere
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the sphere
 */
Transformation getSphericalPosition(const Vector3f& position,
                                    const float radius, const size_t occurence,
                                    const size_t occurences,
                                    const RandomizationInformation& randInfo);

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
                                 const RandomizationInformation& randInfo);

/**
 * @brief Get the Cubic Position object
 *
 * @param center Center of the cube in the 3D scene
 * @param size Size of the side of cube
 * @param randInfo Type of randomization to apply to the position and
 * rotation
 * @return Transformation of the random position and rotation on the plane
 */
Transformation getCubicPosition(const Vector3f& center, const float size,
                                const RandomizationInformation& randInfo);

/**
 * @brief
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
Transformation getSinosoidalPosition(const Vector3f& center, const float size,
                                     const float amplitude,
                                     const size_t occurence,
                                     const RandomizationInformation& randInfo);

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
                              const RandomizationInformation& randInfo);

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
    const size_t occurences, const RandomizationInformation& randInfo,
    const float morphingStep);

/**
 * @brief Set the default transfer function (Unipolar) to a given model
 *
 * @param model Model to which the transfer function should be set
 */
void setDefaultTransferFunction(Model& model);

/**
 * @brief Get the Clipping Planes from the scene
 *
 * @param scene 3D scene
 * @return Vector4fs List of clipping planes
 */
Vector4fs getClippingPlanes(const Scene& scene);

/**
 * @brief Generate a random quaternion
 *
 * @param seed Seed to apply to the randomness
 * @return Quaterniond Random quaternion
 */
Quaterniond randomQuaternion(const size_t seed);

} // namespace bioexplorer
