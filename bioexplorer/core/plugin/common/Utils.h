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

#include <brayns/common/types.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

/**
 * @brief Left trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& ltrim(std::string& s);

/**
 * @brief Right trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& rtrim(std::string& s);

/**
 * @brief Left and right trim of a string
 *
 * @param s String to trim
 * @return Trimmed string
 */
std::string& trim(std::string& s);

/**
 * @brief isClipped Determine if a 3d position is inside a volume defined by
 * clipping planes
 *
 * @param position Position to check
 * @param clippingPlanes Clipping planes defining the volume
 * @return True if the position does not belong to the volume, false otherwise
 */
bool isClipped(const Vector3d& position, const Vector4ds& clippingPlanes);

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
 * @return Vector4ds List of clipping planes
 */
Vector4ds getClippingPlanes(const Scene& scene);

/**
 * @brief Converts a vector of doubles into a 2D vector
 *
 * @param value Vector of doubles
 * @return Vector2d A 2D vector
 */
Vector2d doublesToVector2d(const doubles& value);

/**
 * @brief Converts a vector of doubles into a 3D vector
 *
 * @param value Vector of doubles
 * @return Vector3d A 3D vector
 */
Vector3d doublesToVector3d(const doubles& value);

/**
 * @brief Converts a vector of doubles into a Quaternion
 *
 * @param values Vector of doubles
 * @return Quaternion A quaternion
 */
Quaterniond doublesToQuaterniond(const doubles& values);

/**
 * @brief Converts a vector of doubles into vector of 4D vectors
 *
 * @param values Vector of doubles
 * @return Quaternion A vector of 4D vectors
 */
Vector4ds doublesToVector4ds(const doubles& values);

/**
 * @brief Converts a vector of doubles into randomization details
 *
 * @param value Vector of doubles
 * @return AnimationDetails The randomization details
 */
AnimationDetails doublesToAnimationDetails(const doubles& values);

void sphereFilling(const double radius, const uint64_t occurrence,
                   const uint64_t occurrences, Vector3d& position,
                   Quaterniond& rotation, const double radiusOffset,
                   const double ratio = 1.0);

std::vector<std::string> split(const std::string& s,
                               const std::string& delimiter);

Transformation combineTransformations(const Transformations& transformations);

Quaterniond safeQuatlookAt(const Vector3d& v);

bool rayBoxIntersection(const Vector3d& origin, const Vector3d& direction,
                        const Boxd& box, const double t0, const double t1,
                        double& t);

} // namespace common
} // namespace bioexplorer
