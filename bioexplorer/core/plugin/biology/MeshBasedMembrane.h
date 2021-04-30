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

#include <plugin/biology/Node.h>

#include <brayns/engineapi/Model.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

namespace bioexplorer
{
namespace biology
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
    MeshBasedMembrane(Scene& scene, const MeshBasedMembraneDetails& details);

    /**
     * @brief Destroy the Mesh Based Membrane object
     *
     */
    ~MeshBasedMembrane();

    /**
     * @brief Get the list of proteins defining the membrane
     *
     * @return const ProteinMap& list of proteins defining the membrane
     */
    const ProteinMap& getProteins() const { return _proteins; }

private:
    void _processInstances(const Vector3f& proteinsAverageSize);
    float _getSurfaceArea(const Vector3f& a, const Vector3f& b,
                          const Vector3f& c) const;
    Vector3f _toVector3f(const aiVector3D& v) const;
    Vector3f _toVector3f(const aiVector3D& v, const Vector3f& center,
                         const Vector3f& scaling,
                         const Quaterniond& rotation) const;
    const std::string _getElementNameFromId(const size_t id) const;

    Scene& _scene;
    MeshBasedMembraneDetails _details;
    ProteinMap _proteins;
};
} // namespace biology
} // namespace bioexplorer
