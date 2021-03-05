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

#include "MeshBasedMembrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/io/MeshLoader.h>

namespace bioexplorer
{
MeshBasedMembrane::MeshBasedMembrane(Scene& scene,
                                     const MeshBasedMembraneDescriptor& md)
    : Node()
{
    const auto loader = MeshLoader(scene);

    // Load protein
    const Vector3f position = {md.position[0], md.position[1], md.position[2]};
    const Quaterniond orientation = {md.orientation[0], md.orientation[1],
                                     md.orientation[2], md.orientation[3]};
    const Vector3f scale = {md.scale[0], md.scale[1], md.scale[2]};

    ProteinDescriptor pd;
    pd.assemblyName = md.assemblyName;
    pd.name = md.name;
    pd.contents = md.proteinContents;
    pd.representation = ProteinRepresentation::atoms;
    pd.atomRadiusMultiplier = 1.f;
    pd.recenter = true;
    pd.atomRadiusMultiplier = md.atomRadiusMultiplier;
    pd.representation = md.representation;
    pd.position = md.position;
    pd.orientation = md.orientation;

    // Random seed
    srand(md.randomSeed);

    // Create model
    _protein = ProteinPtr(new Protein(scene, pd));
    _modelDescriptor = _protein->getModelDescriptor();
    auto model = &_modelDescriptor->getModel();
    model->updateBounds();
    const auto proteinSize = model->getBounds().getSize();

    // Load MeshBasedMembrane
    Assimp::Importer importer;
    const aiScene* aiScene =
        importer.ReadFileFromMemory(md.meshContents.c_str(),
                                    md.meshContents.length(),
                                    aiProcess_GenSmoothNormals |
                                        aiProcess_Triangulate);

    if (!aiScene)
        PLUGIN_THROW(std::runtime_error(importer.GetErrorString()));

    if (!aiScene->HasMeshes())
        PLUGIN_THROW(std::runtime_error("No MeshBasedMembranees found"));

    const auto trfm = aiScene->mRootNode->mTransformation;
    const Matrix4f matrix{trfm.a1, trfm.b1, trfm.c1, trfm.d1, trfm.a2, trfm.b2,
                          trfm.c2, trfm.d2, trfm.a3, trfm.b3, trfm.c3, trfm.d3,
                          trfm.a4, trfm.b4, trfm.c4, trfm.d4};

    // Add protein instances according to MeshBasedMembrane topology
    size_t instanceCount = 0;
    float MeshBasedMembraneCoveringProgress = 0.f;
    float instanceCoveringProgress = 0.f;
    for (size_t m = 0; m < aiScene->mNumMeshes; ++m)
    {
        const auto& MeshBasedMembrane = aiScene->mMeshes[m];

        // MeshBasedMembrane scaling
        Vector3f MeshBasedMembraneCenter{0.f, 0.f, 0.f};
        for (size_t i = 0; i < MeshBasedMembrane->mNumVertices; ++i)
        {
            const auto& v = MeshBasedMembrane->mVertices[i];
            MeshBasedMembraneCenter += _toVector3f(v);
        }
        MeshBasedMembraneCenter /= MeshBasedMembrane->mNumVertices;

        // Compute full MeshBasedMembrane area
        std::vector<Vector3ui> faces;
        for (size_t f = 0; f < MeshBasedMembrane->mNumFaces; ++f)
            if (MeshBasedMembrane->mFaces[f].mNumIndices == 3)
                faces.push_back(
                    Vector3ui(MeshBasedMembrane->mFaces[f].mIndices[0],
                              MeshBasedMembrane->mFaces[f].mIndices[1],
                              MeshBasedMembrane->mFaces[f].mIndices[2]));

        float MeshBasedMembraneSurface = 0.f;
        for (const auto& face : faces)
            MeshBasedMembraneSurface += _getSurfaceArea(
                _toVector3f(MeshBasedMembrane->mVertices[face.x],
                            MeshBasedMembraneCenter, scale),
                _toVector3f(MeshBasedMembrane->mVertices[face.y],
                            MeshBasedMembraneCenter, scale),
                _toVector3f(MeshBasedMembrane->mVertices[face.z],
                            MeshBasedMembraneCenter, scale));

        const float proteinSurface = proteinSize.x * proteinSize.x;

        // Total number of instance needed to fill the MeshBasedMembrane surface
        const size_t nbInstances =
            md.density * MeshBasedMembraneSurface / proteinSurface;
        const float instanceSurface = MeshBasedMembraneSurface / nbInstances;

        PLUGIN_INFO << "----===  MeshBasedMembrane  ===----" << std::endl;
        PLUGIN_INFO << "Number of faces      : " << faces.size() << std::endl;
        PLUGIN_INFO << "MeshBasedMembrane surface area    : "
                    << MeshBasedMembraneSurface << std::endl;
        PLUGIN_INFO << "Protein surface area : " << proteinSurface << std::endl;
        PLUGIN_INFO << "Instance surface area: " << instanceSurface
                    << std::endl;
        PLUGIN_INFO << "Number of instances  : " << nbInstances << std::endl;

        for (const auto& face : faces)
        {
            const auto P0 = _toVector3f(MeshBasedMembrane->mVertices[face.x],
                                        MeshBasedMembraneCenter, scale);
            const auto P1 = _toVector3f(MeshBasedMembrane->mVertices[face.y],
                                        MeshBasedMembraneCenter, scale);
            const auto P2 = _toVector3f(MeshBasedMembrane->mVertices[face.z],
                                        MeshBasedMembraneCenter, scale);

            const auto V0 = P1 - P0;
            const auto V1 = P2 - P0;

            const Vector3f defaultNormal = glm::cross(V0, V1);

            // Compute face surface
            const float faceSurface = _getSurfaceArea(P0, P1, P2);

            // Estimate number of proteins for current face
            MeshBasedMembraneCoveringProgress += faceSurface;
            const size_t nbProteins = size_t(
                (MeshBasedMembraneCoveringProgress - instanceCoveringProgress) /
                instanceSurface);

            // compute protein positions and orientations
            for (size_t i = 0; i < nbProteins; ++i)
            {
                instanceCoveringProgress += instanceSurface;
                Vector2f coordinates{1.f, 1.f};
                while (coordinates.x + coordinates.y > 1.f)
                {
                    coordinates.x = float(rand() % nbProteins) / nbProteins;
                    coordinates.y = float(rand() % nbProteins) / nbProteins;
                }

                Transformation tf;
                const Vector3f P = P0 + V0 * coordinates.x + V1 * coordinates.y;
                const Vector3f transformedVertex =
                    matrix * Vector4f(P.x, P.y, P.z, 1.f);

                const Vector3f position{md.position[0], md.position[1],
                                        md.position[2]};

                const float variableOffset =
                    md.surfaceVariableOffset * (rand() % 1000 / 1000.f - 0.5f);

                tf.setTranslation(position + transformedVertex +
                                  defaultNormal * md.surfaceFixedOffset +
                                  defaultNormal * variableOffset);

                if (MeshBasedMembrane->HasNormals())
                {
                    const auto v0 = P0 - P;
                    const auto v1 = P1 - P;
                    const auto v2 = P2 - P;

                    const Vector3f areas{0.5f * length(glm::cross(v1, v2)),
                                         0.5f * length(glm::cross(v0, v2)),
                                         0.5f * length(glm::cross(v0, v1))};

                    const auto N0 =
                        _toVector3f(MeshBasedMembrane->mNormals[face.x]);
                    const auto N1 =
                        _toVector3f(MeshBasedMembrane->mNormals[face.y]);
                    const auto N2 =
                        _toVector3f(MeshBasedMembrane->mNormals[face.z]);

                    const Vector3f normal = glm::normalize(
                        matrix *
                        Vector4f(glm::normalize((N0 * areas.x + N1 * areas.y +
                                                 N2 * areas.z) /
                                                (areas.x + areas.y + areas.z)),
                                 1.f));

                    if (normal != UP_VECTOR)
                    {
                        const Quaterniond orientation{md.orientation[0],
                                                      md.orientation[1],
                                                      md.orientation[2],
                                                      md.orientation[3]};
                        const Quaterniond rotation =
                            glm::quatLookAt(normal, UP_VECTOR);
                        tf.setRotation(rotation * orientation);
                    }
                    tf.setTranslation(position + transformedVertex +
                                      normal * md.surfaceFixedOffset +
                                      normal * variableOffset);
                }

                if (instanceCount == 0)
                    _modelDescriptor->setTransformation(tf);

                const ModelInstance instance(true, false, tf);
                _modelDescriptor->addInstance(instance);

                ++instanceCount;
            }
        }
    }
}

float MeshBasedMembrane::_getSurfaceArea(const Vector3f& v0, const Vector3f& v1,
                                         const Vector3f& v2) const
{
    // Compute triangle area
    const float a = length(v1 - v0);
    const float b = length(v2 - v1);
    const float c = length(v0 - v2);
    const float p = (a + b + c) / 2.f;
    const float e = p * (p - a) * (p - b) * (p - c);
    if (e < 0)
        return 0.f;

    return sqrt(e);
}

Vector3f MeshBasedMembrane::_toVector3f(const aiVector3D& v) const
{
    return Vector3f(v.x, v.y, v.z);
}

Vector3f MeshBasedMembrane::_toVector3f(const aiVector3D& v,
                                        const Vector3f& center,
                                        const Vector3f& scale) const
{
    const Vector3f p{v.x, v.y, v.z};
    const Vector3f a = p - center;
    const Vector3f b = p + a * scale;
    return b;
}

} // namespace bioexplorer
