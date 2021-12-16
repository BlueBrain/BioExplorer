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
    MeshShape(const Vector3f& scale, const Vector4fs& clippingPlanes,
              const std::string& contents);

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere
     *
     * @param occurence Occurence of the position amongst the maximum of
     * occurences (see next parameters)
     * @return Transformation of the random position and rotation on the fan
     */
    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset) const final;

    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset,
                                     const float morphingStep) const final;

    bool isInside(const Vector3f& point) const final;

private:
    float _getSurfaceArea(const Vector3f& a, const Vector3f& b,
                          const Vector3f& c) const;

    Vector3f _toVector3f(const aiVector3D& v) const;
    Vector3f _toVector3f(const aiVector3D& v, const Vector3f& center,
                         const Vector3f& scale) const;
    Vector3f _toVector3f(const aiVector3D& v, const Vector3f& center,
                         const Vector3f& scale,
                         const Quaterniond& rotation) const;

    bool _rayBoxIntersection(const Vector3f& origin, const Vector3f& direction,
                             const Boxf& box, const float t0,
                             const float t1) const;

    std::vector<Vector3ui> _faces;
    floats _faceSurfaces;
    Vector3fs _vertices;
    Vector3fs _normals;
};

} // namespace common
} // namespace bioexplorer
