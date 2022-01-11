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

#include "Shape.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

namespace bioexplorer
{
namespace common
{
using namespace details;
using namespace brayns;

class MeshShape : public Shape
{
public:
    MeshShape(const Vector3d& scale, const Vector4ds& clippingPlanes,
              const std::string& contents);

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere
     *
     * @param occurrence occurrence of the position amongst the maximum of
     * occurrences (see next parameters)
     * @return Transformation of the random position and rotation on the fan
     */
    Transformation getTransformation(const uint64_t occurrence,
                                     const uint64_t nbOccurrences,
                                     const AnimationDetails& animationDetails,
                                     const double offset) const final;

    bool isInside(const Vector3d& point) const final;

private:
    double _getSurfaceArea(const Vector3d& a, const Vector3d& b,
                           const Vector3d& c) const;

    Vector3d _toVector3d(const aiVector3D& v) const;
    Vector3d _toVector3d(const aiVector3D& v, const Vector3d& center,
                         const Vector3d& scale) const;
    Vector3d _toVector3d(const aiVector3D& v, const Vector3d& center,
                         const Vector3d& scale,
                         const Quaterniond& rotation) const;

    std::vector<Vector3ui> _faces;
    doubles _faceSurfaces;
    Vector3ds _vertices;
    Vector3ds _normals;
};

} // namespace common
} // namespace bioexplorer
