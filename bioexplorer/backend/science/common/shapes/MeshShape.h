/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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
