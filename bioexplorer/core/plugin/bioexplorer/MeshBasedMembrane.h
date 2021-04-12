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
 * @brief A MeshBasedMembrane object implements a 3D structure that has the
 * shape of a given OBJ Mesh, but with a surface composed of
 * instances of a protein. This class is typicaly used to create membranes with
 * a shape provided by a 3D Mesh
 *
 */
class MeshBasedMembrane : public Node
{
public:
    /**
     * @brief Construct a new MeshBasedMembrane object
     *
     * @param scene The 3D scene where the glycans are added
     * @param descriptor The data structure describing the MeshBasedMembrane,
     * the protein, and the associated parameters
     */
    MeshBasedMembrane(Scene& scene,
                      const MeshBasedMembraneDescriptor& descriptor);

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
                         const Vector3f& scaling,
                         const Quaterniond& rotation) const;

    ProteinPtr _protein;
};
} // namespace bioexplorer
