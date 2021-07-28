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
 * _details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "MeshBasedMembrane.h"
#include "Protein.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Shapes.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/io/MeshLoader.h>

#include <omp.h>

namespace bioexplorer
{
namespace biology
{
using namespace common;

MeshBasedMembrane::MeshBasedMembrane(Scene& scene,
                                     const Vector3f& assemblyPosition,
                                     const Quaterniond& assemblyRotation,
                                     const Vector4fs& clippingPlanes,
                                     const MeshBasedMembraneDetails& details)
    : Membrane(scene, assemblyPosition, assemblyRotation, clippingPlanes)
    , _details(details)
{
    _processMesh();
}

bool MeshBasedMembrane::isInside(const Vector3f& point) const
{
    const Vector3f& center = _bounds.getCenter();
    const Vector3f rayDirection = center - point;
    const float rayLength = length(rayDirection);
    const Vector3f direction = normalize(rayDirection);
    for (const auto& protein : _proteins)
    {
        const auto modelDescriptor = protein.second->getModelDescriptor();
        const auto& proteinBounds = protein.second->getBounds();
        for (const auto& instance : modelDescriptor->getInstances())
        {
            const auto& tf = instance.getTransformation();
            const Vector3f& translation = tf.getTranslation();
            Boxf box;
            box.merge(translation + proteinBounds.getMin());
            box.merge(translation + proteinBounds.getMax());
            if (_rayBoxIntersection(point, direction, box, rayLength / 10.f,
                                    rayLength))
                return false;
        }
    }
    return true;
}

bool MeshBasedMembrane::_rayBoxIntersection(const Vector3f& origin,
                                            const Vector3f& direction,
                                            const Boxf& box, const float t0,
                                            const float t1) const
{
    const Vector3f bounds[2]{box.getMin(), box.getMax()};
    const Vector3f invDir = 1.f / direction;
    const Vector3ui sign{invDir.x < 0.f, invDir.y < 0.f, invDir.z < 0.f};

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[sign.x].x - origin.x) * invDir.x;
    tmax = (bounds[1 - sign.x].x - origin.x) * invDir.x;
    tymin = (bounds[sign.y].y - origin.y) * invDir.y;
    tymax = (bounds[1 - sign.y].y - origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sign.z].z - origin.z) * invDir.z;
    tzmax = (bounds[1 - sign.z].z - origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return (tmin < t1 && tmax > t0);
}

void MeshBasedMembrane::_processMeshAsProteinInstances()
{
    // Random seed
    srand(_details.randomSeed);

    // Load mesh
    const auto loader = MeshLoader(_scene);
    Assimp::Importer importer;
    const aiScene* aiScene =
        importer.ReadFileFromMemory(_details.meshContents.c_str(),
                                    _details.meshContents.length(),
                                    aiProcess_GenSmoothNormals |
                                        aiProcess_Triangulate);

    if (!aiScene)
        PLUGIN_THROW(importer.GetErrorString());

    if (!aiScene->HasMeshes())
        PLUGIN_THROW("No mesh found");

    const auto trfm = aiScene->mRootNode->mTransformation;
    const Matrix4f mainTransformation{trfm.a1, trfm.b1, trfm.c1, trfm.d1,
                                      trfm.a2, trfm.b2, trfm.c2, trfm.d2,
                                      trfm.a3, trfm.b3, trfm.c3, trfm.d3,
                                      trfm.a4, trfm.b4, trfm.c4, trfm.d4};

    // Protein models
    std::vector<std::string> proteinContents;
    proteinContents.push_back(_details.proteinContents1);
    if (!_details.proteinContents2.empty())
        proteinContents.push_back(_details.proteinContents2);
    if (!_details.proteinContents3.empty())
        proteinContents.push_back(_details.proteinContents3);
    if (!_details.proteinContents4.empty())
        proteinContents.push_back(_details.proteinContents4);

    Vector3f proteinsAverageSize;
    size_t i = 0;
    for (const auto& proteinContent : proteinContents)
    {
        ProteinDetails pd;
        pd.assemblyName = _details.assemblyName;
        pd.name = _getElementNameFromId(i);
        pd.contents = proteinContent;
        pd.recenter = true;
        pd.atomRadiusMultiplier = _details.atomRadiusMultiplier;
        pd.representation = _details.representation;
        pd.position = _details.position;
        pd.rotation = _details.rotation;

        // Create model
        ProteinPtr protein(new Protein(_scene, pd));
        proteinsAverageSize += protein->getBounds().getSize();
        _proteins[pd.name] = std::move(protein);
        ++i;
    }
    proteinsAverageSize /= proteinContents.size();

    // Transformation
    const auto membranePosition = floatsToVector3f(_details.position);
    const auto membraneRotation = floatsToQuaterniond(_details.rotation);
    const auto membraneScale = floatsToVector3f(_details.scale);

    // Add protein instances according to membrane topology
    for (size_t m = 0; m < aiScene->mNumMeshes; ++m)
    {
        aiMesh* mesh = aiScene->mMeshes[m];

        // Randomization
        const auto& params = _details.assemblyParams;
        auto randInfo =
            floatsToRandomizationDetails(params, _details.randomSeed,
                                         PositionRandomizationType::radial);

        Vector3f meshCenter{0.f, 0.f, 0.f};
        for (size_t i = 0; i < mesh->mNumVertices; ++i)
        {
            const auto& v = mesh->mVertices[i];
            meshCenter += _toVector3f(v);
        }
        meshCenter /= mesh->mNumVertices;

        // Compute full membrane area
        float meshCoveringProgress = 0.f;
        float instanceCoveringProgress = 0.f;
        std::vector<Vector3ui> faces;
        for (size_t f = 0; f < mesh->mNumFaces; ++f)
            if (mesh->mFaces[f].mNumIndices == 3)
            {
                const auto i1 = mesh->mVertices[mesh->mFaces[f].mIndices[0]];
                const auto v1 = membranePosition +
                                Vector3f(mainTransformation *
                                         Vector4f(_toVector3f(i1, meshCenter,
                                                              membraneScale),
                                                  1.f));
                const auto i2 = mesh->mVertices[mesh->mFaces[f].mIndices[1]];
                const auto v2 = membranePosition +
                                Vector3f(mainTransformation *
                                         Vector4f(_toVector3f(i2, meshCenter,
                                                              membraneScale),
                                                  1.f));

                const auto i3 = mesh->mVertices[mesh->mFaces[f].mIndices[2]];
                const auto v3 = membranePosition +
                                Vector3f(mainTransformation *
                                         Vector4f(_toVector3f(i3, meshCenter,
                                                              membraneScale),
                                                  1.f));

                faces.push_back(Vector3ui(mesh->mFaces[f].mIndices[0],
                                          mesh->mFaces[f].mIndices[1],
                                          mesh->mFaces[f].mIndices[2]));
            }

        float meshSurface = 0.f;
        for (const auto& face : faces)
            meshSurface += _getSurfaceArea(
                _toVector3f(mesh->mVertices[face.x], meshCenter, membraneScale),
                _toVector3f(mesh->mVertices[face.y], meshCenter, membraneScale),
                _toVector3f(mesh->mVertices[face.z], meshCenter,
                            membraneScale));

        const float proteinSurface =
            proteinsAverageSize.x * proteinsAverageSize.x;

        // Total number of instance needed to fill the membrane surface
        const size_t nbInstances =
            _details.density * meshSurface / proteinSurface;
        const float instanceSurface = meshSurface / nbInstances;

        std::map<size_t, size_t> instanceCounts;
        for (size_t i = 0; i < _proteins.size(); ++i)
            instanceCounts[i] = 0;

        for (const auto& face : faces)
        {
            const auto P0 =
                _toVector3f(mesh->mVertices[face.x], meshCenter, membraneScale);
            const auto P1 =
                _toVector3f(mesh->mVertices[face.y], meshCenter, membraneScale);
            const auto P2 =
                _toVector3f(mesh->mVertices[face.z], meshCenter, membraneScale);

            const auto V0 = P1 - P0;
            const auto V1 = P2 - P0;

            const Vector3f defaultNormal = glm::cross(V0, V1);

            // Compute face surface
            const float faceSurface = _getSurfaceArea(P0, P1, P2);

            // Estimate number of proteins for current face
            meshCoveringProgress += faceSurface;
            const size_t nbProteins = std::max(0.f, (meshCoveringProgress -
                                                     instanceCoveringProgress) /
                                                        instanceSurface);

            // Compute protein positions and rotations
            for (size_t i = 0; i < nbProteins; ++i)
            {
                instanceCoveringProgress += instanceSurface;
                Vector2f coordinates{1.f, 1.f};
                while (coordinates.x + coordinates.y > 1.f)
                {
                    coordinates.x = 0.5f + rnd1();
                    coordinates.y = 0.5f + rnd1();
                }

                Transformation tf;
                const Vector3f P = P0 + V0 * coordinates.x + V1 * coordinates.y;
                const Vector3f transformedVertex =
                    membraneRotation *
                    Vector3d(mainTransformation * Vector4f(P.x, P.y, P.z, 1.f));

                float variableOffset = _details.surfaceVariableOffset;
                if (randInfo.positionSeed != 0)
                    variableOffset += randInfo.positionStrength *
                                      rnd3((randInfo.positionSeed + i) * 10);

                auto translation = membranePosition + transformedVertex +
                                   defaultNormal * _details.surfaceFixedOffset +
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
                        mainTransformation *
                        Vector4f(glm::normalize((N0 * areas.x + N1 * areas.y +
                                                 N2 * areas.z) /
                                                (areas.x + areas.y + areas.z)),
                                 0.f));

                    if (normal != UP_VECTOR)
                    {
                        Quaterniond rotation =
                            glm::quatLookAt(normal, UP_VECTOR);
                        if (randInfo.rotationSeed != 0)
                            rotation = weightedRandomRotation(
                                randInfo.rotationSeed, i, rotation,
                                randInfo.rotationStrength);
                        tf.setRotation(membraneRotation * rotation);
                    }
                    translation = membranePosition + transformedVertex +
                                  normal * _details.surfaceFixedOffset +
                                  normal * variableOffset;
                }

                // Clipping planes
                if (isClipped(translation, _clippingPlanes))
                    continue;

                // Instance
                const size_t id = i % _proteins.size();
                auto protein = _proteins[_getElementNameFromId(id)];
                auto modelDescriptor = protein->getModelDescriptor();

                tf.setTranslation(translation);
                if (instanceCounts[id] == 0)
                    modelDescriptor->setTransformation(tf);

                const ModelInstance instance(true, false, tf);
                modelDescriptor->addInstance(instance);

                instanceCounts[id] = instanceCounts[id] + 1;

                // Bounds
                _bounds.merge(translation);
            }
        }

        PLUGIN_INFO("----===  MeshBasedMembrane  ===----");
        PLUGIN_INFO("Position             : " << membranePosition);
        PLUGIN_INFO("Rotation             : " << membraneRotation);
        PLUGIN_INFO("Scale                : " << membraneScale);
        PLUGIN_INFO("Number of faces      : " << faces.size());
        PLUGIN_INFO("Mesh surface area    : " << meshSurface);
        PLUGIN_INFO("Protein size         : " << proteinsAverageSize);
        PLUGIN_INFO("Protein surface area : " << proteinSurface);
        PLUGIN_INFO("Instance surface area: " << instanceSurface);
        PLUGIN_INFO("Number of instances  : " << nbInstances);
        PLUGIN_INFO("Bounds               : " << _bounds);
    }

    // Add proteins to the scene
    for (size_t i = 0; i < proteinContents.size(); ++i)
        _scene.addModel(
            _proteins[_getElementNameFromId(i)]->getModelDescriptor());
}

void MeshBasedMembrane::_processMeshAsTriangles()
{
    const auto loader = MeshLoader(_scene);
    const auto len = _details.meshContents.length();
    uint8_ts data(len);
    for (size_t i = 0; i < len; ++i)
        data.push_back(_details.meshContents[i]);
    Blob blob = {"obj", _details.assemblyName, data};
    _modelDescriptor =
        loader.importFromBlob(std::move(blob), LoaderProgress(), PropertyMap());
    if (_modelDescriptor)
    {
        // Transformation
        const auto membranePosition = floatsToVector3f(_details.position);
        const auto membraneRotation = floatsToQuaterniond(_details.rotation);
        const auto membraneScale = 2.f * floatsToVector3f(_details.scale);

        Transformation transformation;
        transformation.setTranslation(membranePosition);
        transformation.setRotation(membraneRotation);
        transformation.setScale(membraneScale);

        _modelDescriptor->setTransformation(transformation);

        _scene.addModel(_modelDescriptor);

        // Bounds
        _bounds.merge(membranePosition);
    }
}

void MeshBasedMembrane::_processMesh()
{
    if (_details.proteinContents1.empty() &&
        _details.proteinContents2.empty() &&
        _details.proteinContents3.empty() && _details.proteinContents4.empty())
        _processMeshAsTriangles();
    else
        _processMeshAsProteinInstances();
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

std::string MeshBasedMembrane::_getElementNameFromId(const size_t id)
{
    return _details.assemblyName + "_Membrane_" + std::to_string(id);
}
} // namespace biology
} // namespace bioexplorer
