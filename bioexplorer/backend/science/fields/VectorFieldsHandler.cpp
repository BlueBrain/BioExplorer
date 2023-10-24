/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "VectorFieldsHandler.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

#include <platform/core/common/octree/VectorOctree.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/parameters/ParametersManager.h>

#ifdef USE_OPTIX6
#include <platform/engines/optix6/OptiXVolume.h>
#endif

#include <fstream>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace io;

namespace fields
{
VectorFieldsHandler::VectorFieldsHandler(Engine& engine, core::Model& model, const double voxelSize,
                                         const double density, const uint32_ts& modelIds)
    : FieldsHandler(engine, model, voxelSize, density, modelIds)
{
}

AbstractSimulationHandlerPtr VectorFieldsHandler::clone() const
{
    return std::make_shared<VectorFieldsHandler>(*this);
}

void VectorFieldsHandler::_buildOctree()
{
    PLUGIN_INFO(3, "Building Vector Octree");

    auto& scene = _engine.getScene();
    const auto& clipPlanes = getClippingPlanes(scene);

    OctreeVectors vectors;
    uint32_t count{0};
    const uint32_t densityRatio = 1.f / _density;

    Boxd bounds;
    const auto& modelDescriptors = scene.getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        if (!_modelIds.empty())
        {
            const auto modelId = modelDescriptor->getModelID();
            const auto it = std::find(_modelIds.begin(), _modelIds.end(), modelId);
            if (it == _modelIds.end())
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
    const VectorOctree accelerator(vectors, _voxelSize, minAABB, maxAABB);
    const auto& indices = accelerator.getFlatIndices();
    const auto& data = accelerator.getFlatData();

    const uint32_t volumeSize = accelerator.getVolumeSize();
    _offset = center - extendedHalfSize;
    _dimensions = accelerator.getVolumeDimensions();
    _spacing = sceneSize / Vector3f(_dimensions);

    const auto& params = _engine.getParametersManager().getApplicationParameters();
    const auto& engineName = params.getEngine();
#ifdef USE_OSPRAY
    if (engineName == ENGINE_OSPRAY)
    {
        _frameData.push_back(_offset.x);
        _frameData.push_back(_offset.y);
        _frameData.push_back(_offset.z);
        _frameData.push_back(_spacing.x);
        _frameData.push_back(_spacing.y);
        _frameData.push_back(_spacing.z);
        _frameData.push_back(_dimensions.x);
        _frameData.push_back(_dimensions.y);
        _frameData.push_back(_dimensions.z);
        _frameData.push_back(accelerator.getOctreeSize());
        _frameData.push_back(indices.size());
        _frameData.insert(_frameData.end(), indices.begin(), indices.end());
        _frameData.insert(_frameData.end(), data.begin(), data.end());
        _frameSize = _frameData.size();

        const size_t materialId = 0;
        auto material = _model->createMaterial(0, "Octree");
        const TriangleMesh mesh = createBox(bounds.getMin(), bounds.getMax());
        _model->getTriangleMeshes()[materialId] = mesh;
        _model->markInstancesDirty();
    }
#endif

#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        auto volume = _model->createSharedDataVolume(_dimensions, _spacing, DataType::FLOAT);
        auto optixVolume = dynamic_cast<core::engine::optix::OptiXVolume*>(volume.get());
        optixVolume->setOctree(_offset, indices, data, OctreeDataType::vector);
        _frameData.clear();
        _frameSize = 0;
    }
#endif

    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Vector Octree information (" << vectors.size() << " vectors)");
    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Scene AABB        : " << bounds);
    PLUGIN_INFO(1, "Scene dimension   : " << sceneSize);
    PLUGIN_INFO(1, "Element spacing   : " << _spacing);
    PLUGIN_INFO(1, "Volume dimensions : " << _dimensions);
    PLUGIN_INFO(1, "Element offset    : " << _offset);
    PLUGIN_INFO(1, "Volume size       : " << volumeSize << " bytes");
    PLUGIN_INFO(1, "Indices size      : " << indices.size());
    PLUGIN_INFO(1, "Data size         : " << _frameSize);
    PLUGIN_INFO(1, "Octree size       : " << accelerator.getOctreeSize());
    PLUGIN_INFO(1, "Octree depth      : " << accelerator.getOctreeDepth());
    PLUGIN_INFO(1, "--------------------------------------------");
    _octreeInitialized = true;
}
} // namespace fields
} // namespace bioexplorer
