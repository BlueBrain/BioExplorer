/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include <platform/engines/ospray/ispc/render/utils/SimulationRenderer.h>

namespace sonataexplorer
{
using namespace core;

/**
 * @brief The CellGrowthRenderer class can perform fast transparency
 * and mapping of simulation data on the geometry
 */
class CellGrowthRenderer : public SimulationRenderer
{
public:
    CellGrowthRenderer();

    /**
       Returns the class name as a string
       @return string containing the full name of the class
    */
    std::string toString() const final { return "CellGrowthRenderer"; }
    void commit() final;

private:
    float _simulationThreshold{0.f};

    float _shadows{0.f};
    float _softShadows{0.f};
    float _shadowDistance{1e6f};

    bool _useTransferFunctionColor{false};
};

} // namespace sonataexplorer
