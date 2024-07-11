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

#include <ospray/SDK/render/Renderer.h>

namespace metabolism
{
namespace rendering
{
/**
 * @brief The MetabolismRenderer class allows visualization of atom Metabolism
 * in the 3D scene
 */
class MetabolismRenderer : public ::ospray::Renderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Metabolism Renderer object
     *
     */
    MetabolismRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the name of the object in the OSPRay context
     */
    std::string toString() const final { return "bio_explorer_Metabolism"; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    void commit() final;

private:
    // Shading attributes
    std::vector<void*> _lightArray;
    void** _lightPtr;
    ::ospray::Data* _lightData;

    ::ospray::Material* _bgMaterial;

    float _exposure{1.f};

    float _nearPlane{100.f};
    float _farPlane{1.f};
    float _rayStep{0.1f};
    ::ospray::uint32 _refinementSteps;
    float _alphaCorrection{1.f};
    float _noiseFrequency{1.f};
    float _noiseAmplitude{1.f};
    bool _colorMapPerRegion{false};

    ::ospray::Ref<ospray::Data> _userData;
    ::ospray::uint64 _userDataSize;
};
} // namespace rendering
} // namespace metabolism
