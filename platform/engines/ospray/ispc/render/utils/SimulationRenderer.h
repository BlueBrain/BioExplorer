/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

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

#include "AbstractRenderer.h"

// OSPRay
#include <ospray/SDK/common/Material.h>
#include <ospray/SDK/render/Renderer.h>

namespace core
{
namespace engine
{
namespace ospray
{
/**
 * The SimulationRenderer class implements a parent renderer for
 * all BioExplorer renderers that need to render simulation data
 */
class SimulationRenderer : public AbstractRenderer
{
public:
    void commit() override;

protected:
    ::ospray::Model* _secondaryModel{nullptr};
    float _maxDistanceToSecondaryModel{30.f};

    ::ospray::Ref<::ospray::Data> _userData;
    ::ospray::uint64 _simulationDataSize;

    bool _useHardwareRandomizer{false};
    double _epsilonFactor{1.0};

    double _fogThickness{1e6};
    double _fogStart{0.0};

    ::ospray::uint32 _maxRayDepth{3};
    ::ospray::uint32 _randomNumber{0};

    float _alphaCorrection{0.5f};
};
} // namespace ospray
} // namespace engine
} // namespace core