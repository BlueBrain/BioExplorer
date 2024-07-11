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

#include "OptiXRenderer.h"
#include "OptiXContext.h"
#include "OptiXFrameBuffer.h"
#include "OptiXMaterial.h"
#include "OptiXModel.h"
#include "OptiXScene.h"
#include "OptiXTypes.h"
#include "OptiXUtils.h"

#include <platform/core/engineapi/Engine.h>

namespace core
{
namespace engine
{
namespace optix
{
OptiXRenderer::OptiXRenderer(const AnimationParameters& animationParameters,
                             const RenderingParameters& renderingParameters)
    : Renderer(animationParameters, renderingParameters)
{
    _timer.start();
}

void OptiXRenderer::render(FrameBufferPtr frameBuffer)
{
    if (!frameBuffer->getAccumulation() && frameBuffer->numAccumFrames() > 0)
        return;

    // Provide a random seed to the renderer
    ::optix::float4 jitter = {(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
                              (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX};
    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_RENDERER_JITTER]->setFloat(jitter);
    context[CONTEXT_RENDERER_FRAME]->setUint(frameBuffer->numAccumFrames());

    // Render
    frameBuffer->map();
    const auto size = frameBuffer->getSize();
    context->launch(0, size.x, size.y);
    frameBuffer->unmap();

    frameBuffer->markModified();
}

void OptiXRenderer::commit()
{
    auto& scene = _engine->getScene();
    if (!_renderingParameters.isModified() && !scene.isModified() && !isModified())
        return;

    const bool updateMaterials = isModified() || scene.isModified();

    // If renderer or scene has changed we have to go through all materials in the scene and update the renderer.
    if (updateMaterials)
    {
        const auto& currentRendererType = _engine->getRenderer().getCurrentType();
        const auto renderProgram = OptiXContext::get().getRenderer(currentRendererType);

        scene.visitModels(
            [&](Model& model)
            {
                for (const auto& kv : model.getMaterials())
                {
                    auto optixMaterial = dynamic_cast<OptiXMaterial*>(kv.second.get());
                    const bool textured = optixMaterial->isTextured();
                    auto material = optixMaterial->getOptixMaterial();
                    if (material)
                    {
                        const auto program =
                            textured ? renderProgram->closest_hit_textured : renderProgram->closest_hit;
                        material->setClosestHitProgram(0, program);
                        material->setAnyHitProgram(1, renderProgram->any_hit);
                    }
                    else
                        CORE_DEBUG("No OptiX material initialized for core material " + kv.second->getName());
                }
            });
    }

    // Upload common properties
    auto context = OptiXContext::get().getOptixContext();
    const auto bounds = scene.getBounds();
    const auto epsilon = bounds.getSize().x / 1000.f;
    context[CONTEXT_RENDERER_SCENE_EPSILON]->setFloat(epsilon);
    context[CONTEXT_RENDERER_RADIANCE_RAY_TYPE]->setUint(0);
    context[CONTEXT_RENDERER_SHADOW_RAY_TYPE]->setUint(1);
    context[CONTEXT_RENDERER_AMBIENT_LIGHT_COLOR]->setFloat(_backgroundColor.x, _backgroundColor.y, _backgroundColor.z);
    context[CONTEXT_RENDERER_BACKGROUND_COLOR]->setFloat(_backgroundColor.x, _backgroundColor.y, _backgroundColor.z);
    context[CONTEXT_RENDERER_SAMPLES_PER_PIXEL]->setUint(_spp);

    toOptiXProperties(getPropertyMap());
}

void OptiXRenderer::setCamera(CameraPtr camera)
{
    _camera = static_cast<OptiXCamera*>(camera.get());
    assert(_camera);
    markModified();
}

Renderer::PickResult OptiXRenderer::pick(const Vector2f& pickPos)
{
    PickResult result;
    if (_camera)
    {
        auto context = OptiXContext::get().getOptixContext();
        auto& frameBuffer = _engine->getFrameBuffer();
        OptiXFrameBuffer* optixFrameBuffer = dynamic_cast<OptiXFrameBuffer*>(&frameBuffer);
        if (optixFrameBuffer)
        {
            const auto frameSize = optixFrameBuffer->getSize();
            const size_t indexX = static_cast<size_t>(pickPos.x * frameSize.x);
            const size_t indexY = static_cast<size_t>(pickPos.y * frameSize.y);
            const size_t index = indexY * frameSize.x + indexX;
            if (index >= frameSize.x * frameSize.y)
                return result;
            const float* depthBuffer = optixFrameBuffer->getDepthBuffer();
            const float depth = depthBuffer[index];
            if (depth < INFINITY)
            {
                const Vector3f pos = _camera->getPosition();
                const Vector3f dir = normalize(_camera->getTarget() - _camera->getPosition());
                result.pos = pos + dir * depth;
                result.hit = true;
            }
        }
    }
    return result;
}

} // namespace optix
} // namespace engine
} // namespace core