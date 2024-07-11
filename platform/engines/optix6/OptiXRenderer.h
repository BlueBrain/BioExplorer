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
#include "OptiXCamera.h"

#include <platform/core/common/Timer.h>
#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Renderer.h>

namespace core
{
namespace engine
{
namespace optix
{
/**
   OptiX specific renderer

   This object is the OptiX specific implementation of a renderer
*/
class OptiXRenderer : public Renderer
{
public:
    OptiXRenderer(const AnimationParameters& animationParameters, const RenderingParameters& renderingParameters);

    void render(FrameBufferPtr frameBuffer) final;

    void commit() final;

    void setCamera(CameraPtr camera) final;

    PickResult pick(const Vector2f& pickPos) final;

private:
    OptiXCamera* _camera{nullptr};
    Timer _timer;
};
} // namespace optix
} // namespace engine
} // namespace core