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

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

#include <optixu/optixpp_namespace.h>

#include "OptiXCommonStructs.h"

namespace core
{
namespace engine
{
namespace optix
{
/**

   OptiX specific scene

   This object is the OptiX specific implementation of a scene

*/
class OptiXScene : public Scene
{
public:
    OptiXScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
               VolumeParameters& volumeParameters, FieldParameters& fieldParameters);
    ~OptiXScene();

    /** @copydoc Scene::commit */
    void commit() final;

    /** @copydoc Scene::commitLights */
    bool commitLights() final;

    /** @copydoc Scene::createModel */
    ModelPtr createModel() const final;

    /** @copydoc Scene::supportsConcurrentSceneUpdates. */
    bool supportsConcurrentSceneUpdates() const final { return false; }

private:
    void _commitVolumeParameters();
    void _commitFieldParameters();
    void _commitGeometryParameters();
    void _commitClippingPlanes();

    ::optix::Buffer _lightBuffer{nullptr};
    std::vector<BasicLight> _optixLights;
    ::optix::Group _rootGroup{nullptr};

    // Material Lookup tables
    ::optix::Buffer _colorMapBuffer{nullptr};
    ::optix::Buffer _emissionIntensityMapBuffer{nullptr};
    ::optix::TextureSampler _backgroundTextureSampler{nullptr};
    ::optix::TextureSampler _dummyTextureSampler{nullptr};

    // Volumes
    ::optix::Buffer _volumeBuffer;

    // Clipping planes
    ::optix::Buffer _clipPlanesBuffer{nullptr};
};
} // namespace optix
} // namespace engine
} // namespace core