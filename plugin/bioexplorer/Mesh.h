/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
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

#pragma once

#include <plugin/bioexplorer/Node.h>
#include <plugin/common/Types.h>

#include <brayns/engineapi/Model.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

namespace bioexplorer
{
/**
 * @brief A Mesh object implements a 3D structure that has the shape of a given
 * OBJ mesh, but with a surface composed of instances of a protein. This class
 * is typicaly used to create membranes with a shape provided by a 3D mesh
 *
 */
class Mesh : public Node
{
public:
    /**
     * @brief Construct a new Mesh object
     *
     * @param scene The 3D scene where the glycans are added
     * @param descriptor The data structure describing the mesh, the protein,
     * and the associated parameters
     */
    Mesh(Scene& scene, const MeshDescriptor& descriptor);

    /**
     * @brief Get the Protein object
     *
     * @return ProteinPtr The Protein object
     */
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
