/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#pragma once

#include "Shape.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

namespace bioexplorer
{
namespace common
{
class MeshShape : public Shape
{
public:
    /**
     * @brief Construct a new mesh-based shape object
     *
     * @param clippingPlanes Clipping planes to apply to the shape
     * @param scale Scale of the origin mesh
     * @param contents Contents defining the mesh in a format supported by
     * ASSIMP
     */
    MeshShape(const Vector4ds& clippingPlanes, const core::Vector3d& scale, const std::string& contents);

    /** @copydoc Shape::getTransformation */
    core::Transformation getTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const details::MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const final;

    /** @copydoc Shape::isInside */
    bool isInside(const core::Vector3d& point) const final;

private:
    double _getSurfaceArea(const core::Vector3d& a, const core::Vector3d& b, const core::Vector3d& c) const;

    core::Vector3d _toVector3d(const aiVector3D& v) const;
    core::Vector3d _toVector3d(const aiVector3D& v, const core::Vector3d& center, const core::Vector3d& scale) const;
    core::Vector3d _toVector3d(const aiVector3D& v, const core::Vector3d& center, const core::Vector3d& scale,
                               const core::Quaterniond& rotation) const;

    Vector3uis _faces;
    doubles _faceSurfaces;
    Vector3ds _vertices;
    Vector3ds _normals;
};

} // namespace common
} // namespace bioexplorer
