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
#include <science/common/Utils.h>

#include <fstream>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace io;

namespace fields
{

FieldsHandler::FieldsHandler(const Scene& scene, Model& model, const double voxelSize, const double density,
                             const uint32_ts& modelIds)
    : AbstractSimulationHandler()
    , _scene(&scene)
    , _model(&model)
    , _voxelSize(voxelSize)
    , _density(density)
    , _modelIds(modelIds)
{
    if (density > 1.f || density <= 0.f)
        PLUGIN_THROW("Density should be higher > 0 and <= 1");

    // Load simulation information from compartment reports
    _dt = 1.f;
    _nbFrames = 1;
    _unit = "microns";
    _frameSize = 1;
}

FieldsHandler::FieldsHandler(const std::string& filename)
    : AbstractSimulationHandler()
    , _scene(nullptr)
    , _voxelSize(0.0)
    , _density(0.0)
{
    // Import octree from file
    importFromFile(filename);
    _dt = 1.f;
    _nbFrames = 1;
    _unit = "microns";
}

FieldsHandler::FieldsHandler(const FieldsHandler& rhs)
    : AbstractSimulationHandler(rhs)
{
}

FieldsHandler::~FieldsHandler() {}

void* FieldsHandler::getFrameData(const uint32_t frame)
{
    if (!_octreeInitialized)
        _buildOctree();
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
} // namespace fields
} // namespace bioexplorer