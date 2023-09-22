/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "OptiXRenderer.h"
#include "OptiXContext.h"
#include "OptiXFrameBuffer.h"
#include "OptiXMaterial.h"
#include "OptiXModel.h"
#include "OptiXScene.h"
#include "OptiXTypes.h"
#include "OptiXUtils.h"

namespace core
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
    optix::float4 jitter = {(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
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
    if (!_renderingParameters.isModified() && !_scene->isModified() && !isModified())
        return;

    const bool rendererChanged = _renderingParameters.getCurrentRenderer() != _currentRenderer;

    const bool updateMaterials = isModified() || rendererChanged || _scene->isModified();

    // If renderer or scene has changed we have to go through all materials in the scene and update the renderer.
    if (updateMaterials)
    {
        const auto renderProgram = OptiXContext::get().getRenderer(_renderingParameters.getCurrentRenderer());

        _scene->visitModels(
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
    auto bgColor = _renderingParameters.getBackgroundColor();
    const auto samples_per_pixel = _renderingParameters.getSamplesPerPixel();

    const auto bounds = _scene->getBounds();
    const auto epsilon = bounds.getSize().x / 1000.f;
    context[CONTEXT_RENDERER_SCENE_EPSILON]->setFloat(epsilon);
    context[CONTEXT_RENDERER_RADIANCE_RAY_TYPE]->setUint(0);
    context[CONTEXT_RENDERER_SHADOW_RAY_TYPE]->setUint(1);
    context[CONTEXT_RENDERER_AMBIENT_LIGHT_COLOR]->setFloat(bgColor.x, bgColor.y, bgColor.z);
    context[CONTEXT_RENDERER_BACKGROUND_COLOR]->setFloat(bgColor.x, bgColor.y, bgColor.z);
    context[CONTEXT_RENDERER_SAMPLES_PER_PIXEL]->setUint(samples_per_pixel);

    toOptiXProperties(getPropertyMap());
    _currentRenderer = _renderingParameters.getCurrentRenderer();
}

void OptiXRenderer::setCamera(CameraPtr /*camera*/) {}
} // namespace core
