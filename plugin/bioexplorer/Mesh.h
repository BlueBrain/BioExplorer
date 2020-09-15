/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef BIOEXPLORER_MESH_H
#define BIOEXPLORER_MESH_H

#include <plugin/bioexplorer/Node.h>
#include <plugin/common/Types.h>

#include <brayns/engineapi/Model.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

namespace bioexplorer
{
class Mesh : public Node
{
public:
    Mesh(Scene& scene, const MeshDescriptor& descriptor);

    ProteinPtr getProtein() { return _protein; }

private:
    float _getSurfaceArea(const Vector3f& a, const Vector3f& b,
                          const Vector3f& c) const;
    Vector3f _toVector3f(const aiVector3D& v) const;
    Vector3f _toVector3f(const aiVector3D& v, const Vector3f& center,
                         const Vector3f& scaling) const;

    ProteinPtr _protein;
};
} // namespace bioexplorer

#endif // BIOEXPLORER_MESH_H
