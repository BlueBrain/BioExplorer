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

#include <platform/engines/ospray/ispc/render/utils/SimulationRenderer.h>

namespace bioexplorer
{
namespace rendering
{
/**
 * @brief The PathTracingRenderer class is a renderer that processes the
 * rendering of the 3D scene using the path tracing algorithm
 */
class PathTracingRenderer : public core::engine::ospray::SimulationRenderer
{
public:
    PathTracingRenderer();

    /**
       Returns the class name as a string
       @return string containing the name of the object in the OSPRay context
    */
    std::string toString() const final { return RENDERER_PATH_TRACING; }

    void commit() final;

private:
    // Shading attributes
    double _aoWeight{1.f};
    double _aoDistance{100.f};
};
} // namespace rendering
} // namespace bioexplorer
