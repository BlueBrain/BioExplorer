/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <platform/core/common/MathTypes.h>
#include <platform/core/common/Types.h>

#include <glm/gtx/matrix_decompose.hpp>

namespace sonataexplorer
{
// Convertors
core::Vector3f get_translation(const core::Matrix4f& matrix);

core::Vector3d transformVector3d(const core::Vector3f& v, const core::Matrix4f& transformation);

std::vector<uint64_t> GIDsAsInts(const std::string& gids);

// Containers
bool inBox(const core::Vector3f& point, const core::Boxf& box);
core::Vector3f getPointInSphere(const float innerRadius);
core::Vector3fs getPointsInSphere(const size_t nbPoints, const float innerRadius);

// Volumes
float sphereVolume(const float radius);
float cylinderVolume(const float height, const float radius);
float coneVolume(const float height, const float r1, const float r2);
float capsuleVolume(const float height, const float radius);

} // namespace sonataexplorer
