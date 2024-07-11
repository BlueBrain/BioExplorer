/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include <ospray.h>
#include <platform/core/engineapi/Material.h>

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayMaterial : public Material
{
public:
    OSPRayMaterial(const PropertyMap& properties = {}, const bool backgroundMaterial = false)
        : Material(properties)
        , _isBackGroundMaterial(backgroundMaterial)
    {
    }
    ~OSPRayMaterial();

    /** Noop until commit(renderer) is called. */
    void commit() final;

    /** Instance the actual renderer specific object for this material.
        This operation always creates a new ISPC side material.
     */
    void commit(const std::string& renderer);

    OSPMaterial getOSPMaterial() { return _ospMaterial; }

private:
    OSPTexture _createOSPTexture2D(Texture2DPtr texture);
    OSPMaterial _ospMaterial{nullptr};
    bool _isBackGroundMaterial{false};
    std::string _renderer;
};
} // namespace ospray
} // namespace engine
} // namespace core