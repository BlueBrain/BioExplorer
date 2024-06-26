/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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
