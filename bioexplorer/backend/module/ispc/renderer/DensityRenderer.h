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

#include <science/common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/AbstractRenderer.h>

namespace bioexplorer
{
namespace rendering
{

/**
 * @brief The DensityRenderer class allows visualization of atom density in the
 * 3D scene
 */
class DensityRenderer : public ::core::engine::ospray::AbstractRenderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Density Renderer object
     *
     */
    DensityRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the name of the object in the OSPRay context
     */
    std::string toString() const final { return RENDERER_DENSITY; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    void commit() final;

private:
    // Shading attributes
    double _exposure{1.f};

    double _alphaCorrection{1.f};

    double _rayStep;
    double _searchLength;
    double _farPlane;
    ::ospray::uint32 _samplesPerFrame;
};
} // namespace rendering
} // namespace bioexplorer
