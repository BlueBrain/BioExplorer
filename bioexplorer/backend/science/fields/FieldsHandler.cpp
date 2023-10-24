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

FieldsHandler::FieldsHandler(Engine& engine, Model& model, const double voxelSize, const double density,
                             const uint32_ts& modelIds)
    : AbstractSimulationHandler()
    , _engine(engine)
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

FieldsHandler::FieldsHandler(const FieldsHandler& rhs)
    : AbstractSimulationHandler(rhs)
    , _engine(rhs._engine)
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
} // namespace fields
} // namespace bioexplorer