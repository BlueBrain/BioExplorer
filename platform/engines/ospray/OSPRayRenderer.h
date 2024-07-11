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
#include <platform/core/engineapi/Renderer.h>

#include <ospray.h>

#include "OSPRayCamera.h"

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayRenderer : public Renderer
{
public:
    OSPRayRenderer(const AnimationParameters& animationParameters, const RenderingParameters& renderingParameters);
    ~OSPRayRenderer();

    void render(FrameBufferPtr frameBuffer) final;
    void commit() final;
    float getVariance() const final { return _variance; }
    void setCamera(CameraPtr camera) final;

    PickResult pick(const Vector2f& pickPos) final;

    void setClipPlanes(const Planes& planes);

    /**
       Gets the OSPRay implementation of the renderer object
       @return OSPRay implementation of the renderer object
    */
    OSPRenderer impl() { return _renderer; }

private:
    OSPRayCamera* _camera{nullptr};
    OSPRenderer _renderer{nullptr};
    std::atomic<float> _variance{std::numeric_limits<float>::max()};
    std::string _currentOSPRenderer;
    OSPData _currLightsData{nullptr};

    Planes _clipPlanes;

    void _createOSPRenderer();
    void _commitRendererMaterials();
    void _destroyRenderer();
};
} // namespace ospray
} // namespace engine
} // namespace core