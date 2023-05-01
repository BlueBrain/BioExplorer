/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

// obj
#include "SonataExplorerAbstractRenderer.h"

// ospray
#include <ospray/SDK/render/Renderer.h>

// system
#include <vector>

namespace sonataexplorer
{
/**
 * The SonataExplorerSimulationRenderer class implements a parent renderer for
 * all Brayns renderers that need to render simulation data
 */
class SonataExplorerSimulationRenderer : public SonataExplorerAbstractRenderer
{
public:
    void commit() override;

protected:
    ospray::Model* _secondaryModel;
    float _maxDistanceToSecondaryModel{30.f};

    ospray::Ref<ospray::Data> _simulationData;
    ospray::uint64 _simulationDataSize;

    float _alphaCorrection{0.5f};

    float _fogThickness{1e6f};
    float _fogStart{0.f};
};
} // namespace sonataexplorer
