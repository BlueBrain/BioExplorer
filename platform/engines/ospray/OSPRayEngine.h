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

#include <platform/core/engineapi/Engine.h>

namespace core
{
namespace engine
{
namespace ospray
{
/**
 * OSPRay implementation of the ray-tracing engine.
 */
class OSPRayEngine : public Engine
{
public:
    OSPRayEngine(ParametersManager& parametersManager);

    ~OSPRayEngine();

    /** @copydoc Engine::commit */
    void commit() final;

    /** @copydoc Engine::getMinimumFrameSize */
    Vector2ui getMinimumFrameSize() const final;

    FrameBufferPtr createFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                     FrameBufferFormat frameBufferFormat) const final;

    ScenePtr createScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                         VolumeParameters& volumeParameters, FieldParameters& fieldParameters) const final;
    CameraPtr createCamera() const final;
    RendererPtr createRenderer(const AnimationParameters& animationParameters,
                               const RenderingParameters& renderingParameters) const final;

private:
    void _createCameras();
    void _createRenderers();

    bool _useDynamicLoadBalancer{false};
};
} // namespace ospray
} // namespace engine
} // namespace core
