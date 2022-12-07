/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

// obj renderer
#include "AdvancedMaterial.h"

#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

namespace bioexplorer
{
namespace rendering
{
using namespace ospray;

/**
 * The SimulationRenderer class implements a parent renderer for
 * all BioExplorer renderers that need to render simulation data
 */
class SimulationRenderer : public Renderer
{
public:
    void commit() override;

protected:
    ospray::Model* _secondaryModel{nullptr};
    float _maxDistanceToSecondaryModel{30.f};

    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData{nullptr};

    AdvancedMaterial* _bgMaterial{nullptr};

    ospray::Ref<ospray::Data> _simulationData;
    ospray::uint64 _simulationDataSize;

    bool _useHardwareRandomizer{false};
    bool _showBackground{false};

    double _timestamp{0.0};
    double _exposure{1.0};
    double _epsilonFactor{1.0};

    double _fogThickness{1e6};
    double _fogStart{0.0};

    ospray::uint32 _maxBounces{3};
    ospray::uint32 _randomNumber{0};

    float _alphaCorrection{0.5f};
};
} // namespace rendering
} // namespace bioexplorer
