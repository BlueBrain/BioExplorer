/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/SimulationRenderer.h>

namespace sonataexplorer
{
/**
 * @brief The CellGrowthRenderer class can perform fast transparency
 * and mapping of simulation data on the geometry
 */
class CellGrowthRenderer : public core::engine::ospray::SimulationRenderer
{
public:
    CellGrowthRenderer();

    /**
       Returns the class name as a string
       @return string containing the name of the object in the OSPRay context
    */
    std::string toString() const final { return RENDERER_CELL_GROWTH; }
    void commit() final;

private:
    float _simulationThreshold{0.f};

    float _shadows{0.f};
    float _softShadows{0.f};
    float _shadowDistance{1e6f};

    bool _useTransferFunctionColor{false};
};

} // namespace sonataexplorer
