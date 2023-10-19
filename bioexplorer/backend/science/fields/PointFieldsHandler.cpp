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

#include "PointFieldsHandler.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>
#include <science/common/octree/PointOctree.h>

#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Model.h>

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

PointFieldsHandler::PointFieldsHandler(const Scene& scene, core::Model& model, const double voxelSize,
                                       const double density)
    : FieldsHandler(scene, model, voxelSize, density)
{
}

PointFieldsHandler::PointFieldsHandler(const std::string& filename)
    : FieldsHandler(filename)
{
}

AbstractSimulationHandlerPtr PointFieldsHandler::clone() const
{
    return std::make_shared<PointFieldsHandler>(*this);
}

void PointFieldsHandler::_buildOctree()
{
    PLUGIN_INFO(3, "Building Point Octree");

    const auto& clipPlanes = getClippingPlanes(*_scene);

    OctreePoints points;
    uint32_t count{0};
    const uint32_t densityRatio = 1.f / _density;

    Boxd bounds;
    const auto& modelDescriptors = _scene->getModelDescriptors();
    for (const auto modelDescriptor : modelDescriptors)
    {
        const auto& instances = modelDescriptor->getInstances();
        for (const auto& instance : instances)
        {
            const auto& tf = instance.getTransformation();
            const auto& model = modelDescriptor->getModel();
            const auto& spheresMap = model.getSpheres();
            for (const auto& spheres : spheresMap)
            {
                if (spheres.first != BOUNDINGBOX_MATERIAL_ID && spheres.first != SECONDARY_MODEL_MATERIAL_ID)
                    for (const auto& sphere : spheres.second)
                    {
                        const Vector3f center =
                            tf.getTranslation() + tf.getRotation() * (Vector3d(sphere.center) - tf.getRotationCenter());

                        const Vector3d c = center;
                        if (isClipped(c, clipPlanes))
                        {
                            ++count;
                            continue;
                        }

                        if (count % densityRatio == 0)
                        {
                            bounds.merge(center + sphere.radius);
                            bounds.merge(center - sphere.radius);

                            OctreePoint point;
                            point.position = center;
                            point.radius = sphere.radius;
                            point.value = sphere.radius;
                            points.push_back(point);
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
    const PointOctree accelerator(points, _voxelSize, minAABB, maxAABB);
    const uint32_t volumeSize = accelerator.getVolumeSize();
    _offset = center - extendedHalfSize;
    _dimensions = accelerator.getVolumeDimensions();
    _spacing = sceneSize / Vector3f(_dimensions);

    const auto& indices = accelerator.getFlatIndices();
    const auto& data = accelerator.getFlatData();
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

    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Point Octree information (" << points.size() << " points)");
    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Scene AABB        : " << bounds);
    PLUGIN_INFO(1, "Scene dimension   : " << sceneSize);
    PLUGIN_INFO(1, "Element spacing   : " << _spacing);
    PLUGIN_INFO(1, "Volume dimensions : " << _dimensions);
    PLUGIN_INFO(1, "Element offset    : " << _offset);
    PLUGIN_INFO(1, "Volume size       : " << volumeSize << " bytes");
    PLUGIN_INFO(1, "Indices size      : " << indices.size());
    PLUGIN_INFO(1, "Data size         : " << _frameSize);
    PLUGIN_INFO(1, "PointOctree depth      : " << accelerator.getOctreeDepth());
    PLUGIN_INFO(1, "--------------------------------------------");

#ifdef USE_OPTIX6
    auto volume = _model->createSharedDataVolume(_dimensions, _spacing, DataType::FLOAT);
    auto optixVolume = dynamic_cast<core::engine::optix::OptiXVolume*>(volume.get());
    optixVolume->setOctree(_offset, indices, data, OctreeDataType::point);
#endif
    _octreeInitialized = true;
}
} // namespace fields
} // namespace bioexplorer
