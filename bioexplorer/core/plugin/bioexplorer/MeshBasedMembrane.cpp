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
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/io/MeshLoader.h>

#include <omp.h>

namespace bioexplorer
{
MeshBasedMembrane::MeshBasedMembrane(Scene& scene,
                                     const MeshBasedMembraneDetails& descriptor)
    : Node()
{
    const auto loader = MeshLoader(scene);

    // Load protein
    const Vector3f position = {descriptor.position[0], descriptor.position[1],
                               descriptor.position[2]};
    const Quaterniond rotation = {descriptor.rotation[0],
                                  descriptor.rotation[1],
                                  descriptor.rotation[2],
                                  descriptor.rotation[3]};
    const Vector3f scale = {descriptor.scale[0], descriptor.scale[1],
                            descriptor.scale[2]};

    ProteinDetails pd;
    pd.assemblyName = descriptor.assemblyName;
    pd.name = descriptor.name;
    pd.contents = descriptor.proteinContents;
    pd.recenter = true;
    pd.atomRadiusMultiplier = descriptor.atomRadiusMultiplier;
    pd.representation = descriptor.representation;
    pd.position = descriptor.position;
    pd.rotation = descriptor.rotation;

    // Random seed
    srand(descriptor.randomSeed);

    // Create model
    _protein = ProteinPtr(new Protein(scene, pd));
    _modelDescriptor = _protein->getModelDescriptor();
    auto model = &_modelDescriptor->getModel();
    model->updateBounds();
    const auto proteinSize = model->getBounds().getSize();

    // Clipping planes
    const auto clipPlanes = getClippingPlanes(scene);

    // Load MeshBasedMembrane
    Assimp::Importer importer;
    const aiScene* aiScene =
        importer.ReadFileFromMemory(descriptor.meshContents.c_str(),
                                    descriptor.meshContents.length(),
                                    aiProcess_GenSmoothNormals |
                                        aiProcess_Triangulate);

    if (!aiScene)
        PLUGIN_THROW(importer.GetErrorString());

    if (!aiScene->HasMeshes())
        PLUGIN_THROW("No MeshBasedMembranees found");

    const auto trfm = aiScene->mRootNode->mTransformation;
    const Matrix4f matrix{trfm.a1, trfm.b1, trfm.c1, trfm.d1, trfm.a2, trfm.b2,
                          trfm.c2, trfm.d2, trfm.a3, trfm.b3, trfm.c3, trfm.d3,
                          trfm.a4, trfm.b4, trfm.c4, trfm.d4};

    // Add protein instances according to MeshBasedMembrane topology
    size_t instanceCount = 0;
    float meshCoveringProgress = 0.f;
    float instanceCoveringProgress = 0.f;
    for (size_t m = 0; m < aiScene->mNumMeshes; ++m)
    {
        const auto& mesh = aiScene->mMeshes[m];

        // MeshBasedMembrane scaling
        Vector3f meshCenter{0.f, 0.f, 0.f};
        for (size_t i = 0; i < mesh->mNumVertices; ++i)
        {
            const auto& v = mesh->mVertices[i];
            meshCenter += _toVector3f(v);
        }
        meshCenter /= mesh->mNumVertices;

        // Compute full MeshBasedMembrane area
        std::vector<Vector3ui> faces;
        for (size_t f = 0; f < mesh->mNumFaces; ++f)
            if (mesh->mFaces[f].mNumIndices == 3)
            {
                const auto i1 = mesh->mVertices[mesh->mFaces[f].mIndices[0]];
                const auto v1 =
                    position +
                    Vector3f(matrix * Vector4f(_toVector3f(i1, meshCenter,
                                                           scale, rotation),
                                               1.f));
                const auto i2 = mesh->mVertices[mesh->mFaces[f].mIndices[1]];
                const auto v2 =
                    position +
                    Vector3f(matrix * Vector4f(_toVector3f(i2, meshCenter,
                                                           scale, rotation),
                                               1.f));

                const auto i3 = mesh->mVertices[mesh->mFaces[f].mIndices[2]];
                const auto v3 =
                    position +
                    Vector3f(matrix * Vector4f(_toVector3f(i3, meshCenter,
                                                           scale, rotation),
                                               1.f));

                if (!isClipped(v1, clipPlanes) || !isClipped(v2, clipPlanes) ||
                    !isClipped(v3, clipPlanes))
                    faces.push_back(Vector3ui(mesh->mFaces[f].mIndices[0],
                                              mesh->mFaces[f].mIndices[1],
                                              mesh->mFaces[f].mIndices[2]));
            }

        float meshSurface = 0.f;
        for (const auto& face : faces)
            meshSurface +=
                _getSurfaceArea(_toVector3f(mesh->mVertices[face.x], meshCenter,
                                            scale, rotation),
                                _toVector3f(mesh->mVertices[face.y], meshCenter,
                                            scale, rotation),
                                _toVector3f(mesh->mVertices[face.z], meshCenter,
                                            scale, rotation));

        const float proteinSurface = proteinSize.x * proteinSize.x;

        // Total number of instance needed to fill the MeshBasedMembrane surface
        const size_t nbInstances =
            descriptor.density * meshSurface / proteinSurface;
        const float instanceSurface = meshSurface / nbInstances;

        PLUGIN_INFO("----===  MeshBasedMembrane  ===----");
        PLUGIN_INFO("Position             : " << position);
        PLUGIN_INFO("rotation          : " << rotation);
        PLUGIN_INFO("Scale                : " << scale);
        PLUGIN_INFO("Number of faces      : " << faces.size());
        PLUGIN_INFO("Mesh surface area    : " << meshSurface);
        PLUGIN_INFO("Protein size         : " << proteinSize);
        PLUGIN_INFO("Protein surface area : " << proteinSurface);
        PLUGIN_INFO("Instance surface area: " << instanceSurface);
        PLUGIN_INFO("Number of instances  : " << nbInstances);

#pragma omp parallel for
        for (const auto& face : faces)
        {
            const auto P0 = _toVector3f(mesh->mVertices[face.x], meshCenter,
                                        scale, rotation);
            const auto P1 = _toVector3f(mesh->mVertices[face.y], meshCenter,
                                        scale, rotation);
            const auto P2 = _toVector3f(mesh->mVertices[face.z], meshCenter,
                                        scale, rotation);

            const auto V0 = P1 - P0;
            const auto V1 = P2 - P0;

            const Vector3f defaultNormal = glm::cross(V0, V1);

            // Compute face surface
            const float faceSurface = _getSurfaceArea(P0, P1, P2);

            // Estimate number of proteins for current face
            meshCoveringProgress += faceSurface;
            const size_t nbProteins =
                size_t((meshCoveringProgress - instanceCoveringProgress) /
                       instanceSurface);

            // compute protein positions and rotations
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

                const float variableOffset = descriptor.surfaceVariableOffset *
                                             (rand() % 1000 / 1000.f - 0.5f);

                auto translation =
                    position + transformedVertex +
                    defaultNormal * descriptor.surfaceFixedOffset +
                    defaultNormal * variableOffset;

                if (mesh->HasNormals())
                {
                    const auto v0 = P0 - P;
                    const auto v1 = P1 - P;
                    const auto v2 = P2 - P;

                    const Vector3f areas{0.5f * length(glm::cross(v1, v2)),
                                         0.5f * length(glm::cross(v0, v2)),
                                         0.5f * length(glm::cross(v0, v1))};

                    const auto N0 = _toVector3f(mesh->mNormals[face.x]);
                    const auto N1 = _toVector3f(mesh->mNormals[face.y]);
                    const auto N2 = _toVector3f(mesh->mNormals[face.z]);

                    const Vector3f normal = glm::normalize(
                        matrix *
                        Vector4f(glm::normalize((N0 * areas.x + N1 * areas.y +
                                                 N2 * areas.z) /
                                                (areas.x + areas.y + areas.z)),
                                 1.f));

                    if (normal != UP_VECTOR)
                    {
                        const Quaterniond rotation =
                            glm::quatLookAt(normal, UP_VECTOR);
                        tf.setRotation(rotation * rotation);
                    }
                    translation = position + transformedVertex +
                                  normal * descriptor.surfaceFixedOffset +
                                  normal * variableOffset;
                }

                if (isClipped(translation, clipPlanes))
                    continue;

                tf.setTranslation(translation);

                if (instanceCount == 0)
                    _modelDescriptor->setTransformation(tf);

                const ModelInstance instance(true, false, tf);
#pragma omp critical
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
                                        const Vector3f& scale,
                                        const Quaterniond& rotation) const
{
    const Vector3f p{v.x, v.y, v.z};
    const Vector3f a = p - center;
    const Vector3f b = Vector3f(rotation * Vector3d(p + a)) * scale;
    return b;
}

} // namespace bioexplorer
