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

#include "FieldHandler.h"

#include <plugin/common/Logs.h>

#include <brayns/common/scene/ClipPlane.h>
#include <brayns/engineapi/Model.h>

#include <fstream>

namespace fieldrenderer
{
FieldHandler::FieldHandler()
    : brayns::AbstractSimulationHandler()
{
    // Load simulation information from compartment reports
    _dt = 1.f;
    _unit = "milliseconds";
    _nbFrames = _params.nbFrames;

    // Default values
    for (size_t c; c < 100; ++c)
    {
        const size_t index = rand() % 360;
        floats values;
        for (size_t i = 0; i < _nbFrames; ++i)
            values.push_back(std::max(0.0, cos((index + i) * M_PI / 180.0)));

        _cells.push_back(
            {{rand() % 1000 / 100.f - 5.f, rand() % 1000 / 100.f - 5.f, rand() % 1000 / 100.f - 5.f}, values});
    }
    _frameSize = _cells.size() * _nbFloatsPerElement;
    _frameData.resize(_frameSize * _nbFrames);
}

FieldHandler::FieldHandler(const std::string& uri, const std::string& schema, const bool useCompartments)
    : brayns::AbstractSimulationHandler()
    , _connector(new DBConnector(uri, schema, useCompartments))
{
    _dt = 1.f;
    _nbFrames = _params.nbFrames;
    _unit = "ms";
    if (!_connector)
        // Import from file
        importFromFile(uri);
    else
    {
        const SimulationInformation si = _connector->getSimulationInformation();
        _nbFrames = si.nbFrames;
        _duration = si.endTime;
        _dt = si.timeStep;
        _unit = si.timeUnit;
        _boundingBox = _connector->getCircuitBoundingBox();
    }
}

FieldHandler::FieldHandler(const FieldHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
{
}

FieldHandler::~FieldHandler() {}

void* FieldHandler::getFrameData(const uint32_t frame)
{
    if (_currentFrame == frame)
        return 0;
    _currentFrame = frame;

    if (!_connector)
        for (uint64_t i = 0; i < _frameSize; i += _nbFloatsPerElement)
        {
            const uint64_t cellIndex = i / _nbFloatsPerElement;
            _frameData[i + 0] = _cells[cellIndex].position.x;
            _frameData[i + 1] = _cells[cellIndex].position.y;
            _frameData[i + 2] = _cells[cellIndex].position.z;
            _frameData[i + 3] = _cells[cellIndex].voltages[frame];
        }
    else
    {
        _frameData.clear();
        _frameData.push_back(_boundingBox.getMin().x);
        _frameData.push_back(_boundingBox.getMin().y);
        _frameData.push_back(_boundingBox.getMin().z);
        _frameData.push_back(_boundingBox.getMax().x);
        _frameData.push_back(_boundingBox.getMax().y);
        _frameData.push_back(_boundingBox.getMax().z);
        const auto& points = _connector->getSpikingNeurons(frame * _dt, _params.decaySteps * _dt, _params.density);
        for (const auto point : points)
        {
            _frameData.push_back(point.x);
            _frameData.push_back(point.y);
            _frameData.push_back(point.z);
            _frameData.push_back(point.value);
        }
        _frameSize = points.size() * _nbFloatsPerElement;
    }

    PLUGIN_DEBUG("Simulation data size: " << _frameData.size());
    return _frameData.data();
}

void FieldHandler::importFromFile(const std::string& filename)
{
    PLUGIN_INFO("Loading data from file: " << filename);
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.good())
    {
        const std::string msg = "Could not import data from " + filename;
        PLUGIN_THROW(std::runtime_error(msg));
    }

    // Number of simulation frames
    float value;
    file.read((char*)&value, sizeof(float));
    _nbFrames = (uint64_t)value;

    // Number of cells
    file.read((char*)&value, sizeof(float));
    _frameSize = (uint64_t)value;

    PLUGIN_INFO("Loading " << _frameSize << " cell, and " << _nbFrames << " frames");

    // Read cell positions and voltage values
    _cells.resize(_nbFrames);

    floats record(_nbFrames);
    for (uint64_t i = 0; i < _frameSize; ++i)
    {
        float x, y, z;
        CellInfo cellInfo;
        cellInfo.voltages.resize(_nbFrames);
        file.read((char*)&x, sizeof(float));
        file.read((char*)&y, sizeof(float));
        file.read((char*)&z, sizeof(float));
        cellInfo.position = {x, y, z};
        file.read((char*)cellInfo.voltages.data(), _nbFrames * sizeof(float));
        _cells[i] = cellInfo;
    }

    file.close();
}

AbstractSimulationHandlerPtr FieldHandler::clone() const
{
    return std::make_shared<FieldHandler>(*this);
}
} // namespace fieldrenderer
