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

#include "VectorFieldBuilder.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

#include <platform/core/common/octree/VectorOctree.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Field.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/parameters/ParametersManager.h>

#include <fstream>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace io;

namespace fields
{
VectorFieldBuilder::VectorFieldBuilder()
    : FieldBuilder()
{
}

void VectorFieldBuilder::buildOctree(core::Engine& engine, core::Model& model, const double voxelSize,
                                     const double density, const uint32_ts& modelIds)
{
    PLUGIN_INFO(3, "Building Vector Octree");

    auto& scene = engine.getScene();
    const auto& clipPlanes = getClippingPlanes(scene);

    OctreeVectors vectors;
    uint32_t count{0};
    const uint32_t densityRatio = 1.f / density;

    Boxd bounds;
    const auto& modelDescriptors = scene.getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        if (!modelIds.empty())
        {
            const auto modelId = modelDescriptor->getModelID();
            const auto it = std::find(modelIds.begin(), modelIds.end(), modelId);
            if (it == modelIds.end())
                continue;
        }

        const auto& instances = modelDescriptor->getInstances();
        for (const auto& instance : instances)
        {
            const auto& tf = instance.getTransformation();
            const auto& model = modelDescriptor->getModel();
            const auto& conesMap = model.getCones();
            for (const auto& cones : conesMap)
            {
                if (cones.first != BOUNDINGBOX_MATERIAL_ID && cones.first != SECONDARY_MODEL_MATERIAL_ID)
                    for (const auto& cone : cones.second)
                    {
                        const Vector3f center =
                            tf.getTranslation() + tf.getRotation() * (Vector3d(cone.center) - tf.getRotationCenter());
                        const Vector3f up =
                            tf.getTranslation() + tf.getRotation() * (Vector3d(cone.up) - tf.getRotationCenter());

                        const Vector3d c = center;
                        const Vector3d u = center;
                        if (isClipped(c, clipPlanes) || isClipped(u, clipPlanes))
                        {
                            ++count;
                            continue;
                        }

                        if (count % densityRatio == 0)
                        {
                            bounds.merge(center + cone.centerRadius);
                            bounds.merge(center - cone.centerRadius);
                            bounds.merge(up + cone.upRadius);
                            bounds.merge(up - cone.upRadius);

                            OctreeVector vector;
                            vector.position = center;
                            vector.direction = up - center;
                            vectors.push_back(vector);
                        }
                        ++count;
                    }
            }
        }
    }

    // Events bounds
    Vector3f sceneSize = bounds.getSize();
    Vector3f center = bounds.getCenter();
    Vector3f extendedHalfSize = sceneSize * 0.5f;

    // Expand volume by 25%
    const float boundsExpansion = 1.25f;
    bounds.merge(center + extendedHalfSize * boundsExpansion);
    bounds.merge(center - extendedHalfSize * boundsExpansion);
    sceneSize = bounds.getSize();
    center = bounds.getCenter();
    extendedHalfSize = sceneSize * 0.5f;

    // Compute volume information
    const Vector3f minAABB = center - extendedHalfSize;
    const Vector3f maxAABB = center + extendedHalfSize;

    // Build acceleration structure
    const VectorOctree accelerator(vectors, voxelSize, minAABB, maxAABB);
    const auto& indices = accelerator.getFlatIndices();
    const auto& data = accelerator.getFlatData();

    const uint32_t volumeSize = accelerator.getVolumeSize();
    const auto offset = center - extendedHalfSize;
    const auto dimensions = accelerator.getVolumeDimensions();
    const auto spacing = sceneSize / Vector3f(dimensions);

    const auto& params = engine.getParametersManager().getApplicationParameters();
    const auto& engineName = params.getEngine();
    auto field = model.createField(dimensions, spacing, offset, indices, data, OctreeDataType::odt_vectors);
    const size_t materialId = FIELD_MATERIAL_ID;
    model.addField(materialId, field);
    model.createMaterial(materialId, std::to_string(materialId));

    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Vector Octree information (" << vectors.size() << " vectors)");
    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Scene AABB        : " << bounds);
    PLUGIN_INFO(1, "Scene dimension   : " << sceneSize);
    PLUGIN_INFO(1, "Element spacing   : " << spacing);
    PLUGIN_INFO(1, "Volume dimensions : " << dimensions);
    PLUGIN_INFO(1, "Element offset    : " << offset);
    PLUGIN_INFO(1, "Volume size       : " << volumeSize << " bytes");
    PLUGIN_INFO(1, "Indices size      : " << indices.size());
    PLUGIN_INFO(1, "Octree size       : " << accelerator.getOctreeSize());
    PLUGIN_INFO(1, "Octree depth      : " << accelerator.getOctreeDepth());
    PLUGIN_INFO(1, "--------------------------------------------");
}
} // namespace fields
} // namespace bioexplorer
