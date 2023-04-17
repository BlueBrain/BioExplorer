/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "OctreeFieldHandler.h"

#include <plugin/io/octree/Octree.h>

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/common/scene/ClipPlane.h>
#include <brayns/engineapi/Model.h>

#include <fstream>

namespace fieldrenderer
{
OctreeFieldHandler::OctreeFieldHandler(const std::string& uri, const std::string& schema, const bool useCompartments)
    : brayns::AbstractSimulationHandler()
    , _connector(new DBConnector(uri, schema, useCompartments))
{
    const SimulationInformation si = _connector->getSimulationInformation();
    _nbFrames = si.nbFrames;
    _duration = si.endTime;
    _dt = si.timeStep;
    _unit = si.timeUnit;
    _boundingBox = _connector->getCircuitBoundingBox();
}

void OctreeFieldHandler::_buildOctree(const size_t frame)
{
    PLUGIN_INFO("Building Octree");

    _frameData.clear();
    floats events;
    const auto& points = _connector->getSpikingNeurons(frame * _dt, _params.decaySteps * _dt, _params.density);
    for (const auto point : points)
    {
        events.push_back(point.x);
        events.push_back(point.y);
        events.push_back(point.z);
        events.push_back(point.value);
        events.push_back(point.value);
    }

    if (events.empty())
        return;

    const auto box = _connector->getCircuitBoundingBox();

    // Compute volume information
    const glm::vec3 sceneSize = box.getSize();

    // Double AABB size
    glm::vec3 center = box.getCenter();
    _offset = box.getMin();

    const float voxelSize = std::max(sceneSize.x, std::max(sceneSize.y, sceneSize.z)) / 4096.f;

    // Build acceleration structure
    Octree morphoOctree(events, voxelSize, box.getMin(), box.getMax());
    uint64_t volumeSize = morphoOctree.getVolumeSize();
    _dimensions = morphoOctree.getVolumeDim();
    _spacing = sceneSize / glm::vec3(_dimensions);

    PLUGIN_INFO("--------------------------------------------");
    PLUGIN_INFO("Octree information");
    PLUGIN_INFO("--------------------------------------------");
    PLUGIN_INFO("Scene AABB        : [" << box << "]");
    PLUGIN_INFO("Scene dimension   : [" << sceneSize.x << "," << sceneSize.y << "," << sceneSize.z << "]");
    PLUGIN_INFO("Voxel size        : [" << voxelSize << "] ");
    PLUGIN_INFO("Element spacing   : [" << _spacing.x << ", " << _spacing.y << ", " << _spacing.z << "] ");
    PLUGIN_INFO("Volume dimensions : [" << _dimensions.x << ", " << _dimensions.y << ", " << _dimensions.z
                                        << "] = " << volumeSize << " bytes");

    const auto& indices = morphoOctree.getFlatIndexes();
    PLUGIN_INFO("Indices size      : " << indices.size());
    const auto& data = morphoOctree.getFlatData();
    _frameData.push_back(_offset.x);
    _frameData.push_back(_offset.y);
    _frameData.push_back(_offset.z);
    _frameData.push_back(_spacing.x);
    _frameData.push_back(_spacing.y);
    _frameData.push_back(_spacing.z);
    _frameData.push_back(_dimensions.x);
    _frameData.push_back(_dimensions.y);
    _frameData.push_back(_dimensions.z);
    _frameData.push_back(morphoOctree.getOctreeSize());
    _frameData.push_back(indices.size());
    _frameData.insert(_frameData.end(), indices.begin(), indices.end());
    _frameData.insert(_frameData.end(), data.begin(), data.end());
    _frameSize = _frameData.size();
    PLUGIN_INFO("Data size         : " << _frameSize);
    PLUGIN_INFO("--------------------------------------------");

    if (_frameSize > 1073741824)
        PLUGIN_ERROR("Octree size exceeds 32 bits capacity");
}

OctreeFieldHandler::OctreeFieldHandler(const OctreeFieldHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
{
}

OctreeFieldHandler::~OctreeFieldHandler() {}

void* OctreeFieldHandler::getFrameData(const uint32_t frame)
{
    if (_currentFrame == frame)
        return 0;

    _buildOctree(frame);
    _currentFrame = frame;
    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr OctreeFieldHandler::clone() const
{
    return std::make_shared<OctreeFieldHandler>(*this);
}
} // namespace fieldrenderer
