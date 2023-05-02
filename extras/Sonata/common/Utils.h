/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <brayns/common/mathTypes.h>
#include <brayns/common/types.h>

#include <glm/gtx/matrix_decompose.hpp>

namespace sonataexplorer
{
// Convertors
brayns::Vector3f get_translation(const brayns::Matrix4f& matrix);

brayns::Vector3f transformVector3f(const brayns::Vector3f& v,
                                   const brayns::Matrix4f& transformation);

std::vector<uint64_t> GIDsAsInts(const std::string& gids);

// Containers
bool inBox(const brayns::Vector3f& point, const brayns::Boxf& box);
brayns::Vector3f getPointInSphere(const float innerRadius);
brayns::Vector3fs getPointsInSphere(const size_t nbPoints,
                                    const float innerRadius);

// Volumes
float sphereVolume(const float radius);
float cylinderVolume(const float height, const float radius);
float coneVolume(const float height, const float r1, const float r2);
float capsuleVolume(const float height, const float radius);

} // namespace sonataexplorer
