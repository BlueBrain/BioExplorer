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

#include "utils/SimulationRenderer.h"

namespace core
{
namespace engine
{
namespace ospray
{
/**
 * @brief The AdvancedRenderer class is a renderer that can
 * perform global illumination (light shading, shadowIntensity, ambient occlusion, color
 * bleeding, light emission)
 */
class AdvancedRenderer : public SimulationRenderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Renderer object
     *
     */
    AdvancedRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the name of the object in the OSPRay context
     */
    std::string toString() const final { return RENDERER_PROPERTY_TYPE_ADVANCED; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    void commit() final;

private:
    // Shading
    bool _fastPreview{false};
    double _shadows{0.f};
    double _softShadows{0.f};
    ::ospray::uint32 _softShadowsSamples{1};
    double _giStrength{0.f};
    double _giDistance{1e6};
    ::ospray::uint32 _giSamples{1};
    bool _matrixFilter{false};

    // Volumes
    float _volumeSamplingThreshold{1.f};
    ::ospray::int32 _volumeSamplesPerRay{32};
    float _volumeSpecularExponent{10.f};
    float _volumeAlphaCorrection{0.5f};

    // Clip planes
    ::ospray::Ref<::ospray::Data> clipPlanes;
};
} // namespace ospray
} // namespace engine
} // namespace core