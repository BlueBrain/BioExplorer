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

#include "FieldsHandler.h"

#include <science/common/Logs.h>
#include <science/common/Octree.h>
#include <science/common/Utils.h>

#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Model.h>

#include <fstream>

namespace bioexplorer
{
namespace fields
{
using namespace common;

FieldsHandler::FieldsHandler(const Scene& scene, const double voxelSize, const double density)
    : AbstractSimulationHandler()
{
    // Load simulation information from compartment reports
    _dt = 1.f;
    _nbFrames = 1;
    _unit = "microns";
    _frameSize = 1;
    _buildOctree(scene, voxelSize, density);
}

FieldsHandler::FieldsHandler(const std::string& filename)
    : AbstractSimulationHandler()
{
    // Import octree from file
    importFromFile(filename);
    _dt = 1.f;
    _nbFrames = 1;
    _unit = "microns";
}

void FieldsHandler::_buildOctree(const Scene& scene, const double voxelSize, const double density)
{
    PLUGIN_INFO(3, "Building Octree");

    if (density > 1.f || density <= 0.f)
        PLUGIN_THROW("Density should be higher > 0 and <= 1");

    const auto clipPlanes = getClippingPlanes(scene);

    floats events;
    uint32_t count{0};
    const uint32_t densityRatio = 1.f / density;

    Boxd bounds;
    const auto& modelDescriptors = scene.getModelDescriptors();
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

                        events.push_back(center.x);
                        events.push_back(center.y);
                        events.push_back(center.z);
                        events.push_back(sphere.radius);
                        events.push_back(sphere.radius);
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
    const Octree accelerator(events, voxelSize, minAABB, maxAABB);
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
    PLUGIN_INFO(1, "Octree information");
    PLUGIN_INFO(1, "--------------------------------------------");
    PLUGIN_INFO(1, "Scene AABB        : " << bounds);
    PLUGIN_INFO(1, "Scene dimension   : " << sceneSize);
    PLUGIN_INFO(1, "Element spacing   : " << _spacing);
    PLUGIN_INFO(1, "Volume dimensions : " << _dimensions);
    PLUGIN_INFO(1, "Element offset    : " << _offset);
    PLUGIN_INFO(1, "Volume size       : " << volumeSize << " bytes");
    PLUGIN_INFO(1, "Indices size      : " << indices.size());
    PLUGIN_INFO(1, "Data size         : " << _frameSize);
    PLUGIN_INFO(1, "Octree depth      : " << accelerator.getOctreeDepth());
    PLUGIN_INFO(1, "--------------------------------------------");
}

FieldsHandler::FieldsHandler(const FieldsHandler& rhs)
    : AbstractSimulationHandler(rhs)
{
}

FieldsHandler::~FieldsHandler() {}

void* FieldsHandler::getFrameData(const uint32_t frame)
{
    _currentFrame = frame;
    return _frameData.data();
}

void FieldsHandler::exportToFile(const std::string& filename) const
{
    PLUGIN_INFO(3, "Saving octree to file: " << filename);
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good())
        PLUGIN_THROW("Could not export octree to " + filename);

    file.write((char*)&_frameSize, sizeof(uint32_t));
    file.write((char*)_frameData.data(), _frameData.size() * sizeof(double));

    file.close();
}

void FieldsHandler::importFromFile(const std::string& filename)
{
    PLUGIN_INFO(3, "Loading octree from file: " << filename);
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.good())
        PLUGIN_THROW("Could not import octree from " + filename);

    file.read((char*)&_frameSize, sizeof(uint32_t));
    _frameData.resize(_frameSize);
    file.read((char*)_frameData.data(), _frameData.size() * sizeof(double));

    _offset = {_frameData[0], _frameData[1], _frameData[2]};
    _spacing = {_frameData[3], _frameData[4], _frameData[5]};
    _dimensions = {_frameData[6], _frameData[7], _frameData[8]};

    PLUGIN_INFO(3, "Octree: dimensions=" << _dimensions << ", offset=" << _offset << ", spacing=" << _spacing);

    file.close();
}

AbstractSimulationHandlerPtr FieldsHandler::clone() const
{
    return std::make_shared<FieldsHandler>(*this);
}
} // namespace fields
} // namespace bioexplorer
