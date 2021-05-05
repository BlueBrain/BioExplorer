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
bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes);

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
 * @brief Converts a vector of floats into a 3D vector
 *
 * @param value Vector of floats
 * @return Vector3f A 3D vector
 */
Vector3f floatsToVector3f(const floats& value);

/**
 * @brief Converts a vector of floats into a Quaternion
 *
 * @param value Vector of floats
 * @return Quaternion A quaternion
 */
Quaterniond floatsToQuaterniond(const floats& value);

/**
 * @brief Converts a vector of floats into vector of 4D vectors
 *
 * @param value Vector of floats
 * @return Quaternion A vector of 4D vectors
 */
Vector4fs floatsToVector4fs(const floats& value);

} // namespace common
} // namespace bioexplorer
