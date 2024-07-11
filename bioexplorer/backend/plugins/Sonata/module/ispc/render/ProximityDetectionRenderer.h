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
    The ProximityDetectionRenderer uses an algorithm similar to ambient occlusion to identify touches between
   geometries. A color gradient, defined by nearColor and farColor, is computed according to the distance between the
   intersection that is being rendered and the surrounding geometry. nearColor is used when the distance to the
   surrounding geometry is less than 20% of the detection distance. farColor is used otherwise. The detection distance
   defines the maximum distance between the intersection and the surrounding geometry.

    Surrounding geometry is detected by sending random rays from the intersection point of the surface.

    This renderer can be configured using the following entries:
    - detectionDistance: Maximum distance for surrounding geometry detection
    - materialTestEnabled: If true, detection will be disabled for geometry
   that has the same material as the hit surface.
    - spp: Unsigned integer defining the number of samples per pixel
*/
class ProximityDetectionRenderer : public core::engine::ospray::SimulationRenderer
{
public:
    ProximityDetectionRenderer();

    /**
       Returns the class name as a string
       @return string containing the name of the object in the OSPRay context
    */
    std::string toString() const final { return RENDERER_PROXIMITY; }
    /**
       Commits the changes held by the object so that
       attributes become available to the OSPRay rendering engine
    */
    virtual void commit();

private:
    ::ospray::vec3f _nearColor{0.f, 1.f, 0.f};
    ::ospray::vec3f _farColor{1.f, 0.f, 0.f};
    float _detectionDistance{1.f};
    bool _detectionOnDifferentMaterial{true};
    bool _surfaceShadingEnabled{true};
    ::ospray::uint32 _randomNumber{0};
    float _alphaCorrection{0.5f};
};
} // namespace sonataexplorer
