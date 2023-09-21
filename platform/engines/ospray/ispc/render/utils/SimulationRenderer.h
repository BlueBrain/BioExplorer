/*
 * Copyright (c) 2018, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include "AbstractRenderer.h"

// OSPRay
#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

namespace core
{
using namespace ospray;

/**
 * The SimulationRenderer class implements a parent renderer for
 * all BioExplorer renderers that need to render simulation data
 */
class SimulationRenderer : public AbstractRenderer
{
public:
    void commit() override;

protected:
    ospray::Model* _secondaryModel{nullptr};
    float _maxDistanceToSecondaryModel{30.f};

    ospray::Ref<ospray::Data> _userData;
    ospray::uint64 _simulationDataSize;

    bool _useHardwareRandomizer{false};
    bool _showBackground{false};

    double _exposure{1.0};
    double _epsilonFactor{1.0};

    double _fogThickness{1e6};
    double _fogStart{0.0};

    ospray::uint32 _maxBounces{3};
    ospray::uint32 _randomNumber{0};

    float _alphaCorrection{0.5f};
};
} // namespace core
