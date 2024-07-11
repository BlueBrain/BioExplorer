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

#include "Engine.h"

#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Renderer.h>
#include <platform/core/engineapi/Scene.h>

#include <platform/core/common/ImageManager.h>

#include <platform/core/parameters/ParametersManager.h>

namespace core
{
Engine::Engine(ParametersManager& parametersManager)
    : _parametersManager(parametersManager)
{
}

void Engine::commit()
{
    _renderer->commit();
}

void Engine::preRender()
{
    for (auto frameBuffer : _frameBuffers)
    {
        frameBuffer->setAccumulation(_renderer->getAccumulation());
        frameBuffer->setSubsampling(_renderer->getSubsampling());
    }
}

void Engine::render()
{
    for (auto frameBuffer : _frameBuffers)
    {
        _camera->setBufferTarget(frameBuffer->getName());
        _camera->commit();
        _camera->resetModified();
        _renderer->render(frameBuffer);
    }
}

void Engine::postRender()
{
    for (auto frameBuffer : _frameBuffers)
        frameBuffer->incrementAccumFrames();
}

Renderer& Engine::getRenderer()
{
    return *_renderer;
}

bool Engine::continueRendering() const
{
    auto frameBuffer = _frameBuffers[0];
    return _parametersManager.getAnimationParameters().isPlaying() ||
           (frameBuffer->getAccumulation() && (frameBuffer->numAccumFrames() < _renderer->getMaxAccumFrames()));
}

void Engine::addFrameBuffer(FrameBufferPtr frameBuffer)
{
    _frameBuffers.push_back(frameBuffer);
}

void Engine::removeFrameBuffer(FrameBufferPtr frameBuffer)
{
    _frameBuffers.erase(std::remove(_frameBuffers.begin(), _frameBuffers.end(), frameBuffer), _frameBuffers.end());
}

void Engine::clearFrameBuffers()
{
    for (auto frameBuffer : _frameBuffers)
        frameBuffer->clear();
}

void Engine::resetFrameBuffers()
{
    for (auto frameBuffer : _frameBuffers)
        frameBuffer->resetModified();
}

void Engine::addRendererType(const std::string& name, const PropertyMap& properties)
{
    _parametersManager.getRenderingParameters().addRenderer(name);
    getRenderer().setProperties(name, properties);
    _rendererTypes.push_back(name);
}

void Engine::addCameraType(const std::string& name, const PropertyMap& properties)
{
    _parametersManager.getRenderingParameters().addCamera(name);
    getCamera().setProperties(name, properties);
}
} // namespace core
