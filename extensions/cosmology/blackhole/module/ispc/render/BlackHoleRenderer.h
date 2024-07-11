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

#include <platform/engines/ospray/ispc/render/utils/AbstractRenderer.h>

namespace spaceexplorer
{
namespace blackhole
{
class BlackHoleRenderer : public core::engine::ospray::AbstractRenderer
{
public:
    BlackHoleRenderer();

    /**
       Returns the class name as a string
       @return string containing the name of the object in the OSPRay context
    */
    std::string toString() const final { return "blackhole"; }
    void commit() final;

private:
    // Shading attributes
    float _exposure{1.f};
    ::ospray::uint32 _nbDisks;
    bool _grid{false};
    float _diskRotationSpeed{3.0};
    ::ospray::uint32 _diskTextureLayers{12};
    float _blackHoleSize{0.3f};
};
} // namespace blackhole
} // namespace spaceexplorer
