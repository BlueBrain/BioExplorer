/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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
#if 0
#include "OptiXMaterial.h"
#include "OptiXModel.h"
#endif
#include "OptiXScene.h"

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

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

#if 0
    // Provide a random seed to the renderer
    optix::float4 jitter = {(float)rand() / (float)RAND_MAX,
                            (float)rand() / (float)RAND_MAX,
                            (float)rand() / (float)RAND_MAX,
                            (float)rand() / (float)RAND_MAX};
    context["jitter4"]->setFloat(jitter);
    context["frame"]->setUint(frameBuffer->numAccumFrames());
#endif

    // Render
    frameBuffer->map();
    frameBuffer->unmap();

    frameBuffer->markModified();
}

void OptiXRenderer::commit()
{
    if (!_renderingParameters.isModified() && !_scene->isModified() && !isModified())
    {
        return;
    }

#if 0
    const bool rendererChanged =
        _renderingParameters.getCurrentRenderer() != _currentRenderer;

    const bool updateMaterials =
        isModified() || rendererChanged || _scene->isModified();

    // If renderer or scene has changed we have to go through all materials in
    // the scene and update the renderer.
    if (updateMaterials)
    {
        const auto renderProgram = OptiXContext::get().getRenderer(
            _renderingParameters.getCurrentRenderer());

        _scene->visitModels(
            [&](Model& model)
            {
                for (const auto& kv : model.getMaterials())
                {
                    auto optixMaterial =
                        dynamic_cast<OptiXMaterial*>(kv.second.get());
                    const bool textured = optixMaterial->isTextured();

                    optixMaterial->getOptixMaterial()->setClosestHitProgram(
                        0, textured ? renderProgram->closest_hit_textured
                                    : renderProgram->closest_hit);
                    optixMaterial->getOptixMaterial()->setAnyHitProgram(
                        1, renderProgram->any_hit);
                }
            });
    }

    // Upload common properties
    auto context = OptiXContext::get().getOptixContext();
    auto bgColor = _renderingParameters.getBackgroundColor();
    const auto samples_per_pixel = _renderingParameters.getSamplesPerPixel();
    constexpr auto epsilon = 1e-5f;

    context["radianceRayType"]->setUint(0);
    context["shadowRayType"]->setUint(1);
    context["sceneEpsilon"]->setFloat(epsilon);
    context["ambientLightColor"]->setFloat(bgColor.x, bgColor.y, bgColor.z);
    context["bgColor"]->setFloat(bgColor.x, bgColor.y, bgColor.z);
    context["samples_per_pixel"]->setUint(samples_per_pixel);
    context["currentTime"]->setFloat(_timer.elapsed());

    toOptiXProperties(getPropertyMap());
#endif
    auto& state = OptiXContext::getInstance().getState();
    const auto bgColor = _renderingParameters.getBackgroundColor();
    state.params.ambient_light_color = make_float3(bgColor.x, bgColor.y, bgColor.z);

    _currentRenderer = _renderingParameters.getCurrentRenderer();
}

void OptiXRenderer::setCamera(CameraPtr /*camera*/) {}
} // namespace core
