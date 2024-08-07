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

#include <platform/core/common/Logs.h>
#include <platform/core/common/Properties.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Model.h>

#include <platform/engines/ospray/ispc/camera/AnaglyphCamera.h>

#include "OSPRayCamera.h"
#include "OSPRayFrameBuffer.h"
#include "OSPRayMaterial.h"
#include "OSPRayModel.h"
#include "OSPRayProperties.h"
#include "OSPRayRenderer.h"
#include "OSPRayScene.h"
#include "OSPRayUtils.h"

namespace core
{
namespace engine
{
namespace ospray
{
OSPRayRenderer::OSPRayRenderer(const AnimationParameters& animationParameters,
                               const RenderingParameters& renderingParameters)
    : Renderer(animationParameters, renderingParameters)
{
}

OSPRayRenderer::~OSPRayRenderer()
{
    _destroyRenderer();
}

void OSPRayRenderer::_destroyRenderer()
{
    if (_renderer)
        ospRelease(_renderer);
    _renderer = nullptr;
}

void OSPRayRenderer::render(FrameBufferPtr frameBuffer)
{
    auto osprayFrameBuffer = std::static_pointer_cast<OSPRayFrameBuffer>(frameBuffer);
    auto lock = osprayFrameBuffer->getScopeLock();

    _variance = ospRenderFrame(osprayFrameBuffer->impl(), _renderer, OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM);

    osprayFrameBuffer->markModified();
}

void OSPRayRenderer::commit()
{
    const AnimationParameters& ap = _animationParameters;
    const RenderingParameters& rp = _renderingParameters;
    OSPRayScene& scene = static_cast<OSPRayScene&>(_engine->getScene());
    const bool lightsChanged = (_currLightsData != scene.lightData());
    const bool rendererChanged = (_currentOSPRenderer != getCurrentType());

    if (!ap.isModified() && !rp.isModified() && !scene.isModified() && !isModified() && !_camera->isModified() &&
        !lightsChanged && !rendererChanged)
    {
        return;
    }

    if (rendererChanged)
        _createOSPRenderer();

    toOSPRayProperties(*this, _renderer);

    if (lightsChanged || rendererChanged)
    {
        ospSetData(_renderer, RENDERER_PROPERTY_LIGHTS, scene.lightData());
        _currLightsData = scene.lightData();
    }

    if (isModified() || rendererChanged || scene.isModified())
    {
        _commitRendererMaterials();

        if (auto simulationModel = scene.getSimulatedModel())
        {
            auto& model = static_cast<OSPRayModel&>(simulationModel->getModel());
            ospSetObject(_renderer, RENDERER_PROPERTY_SECONDARY_MODEL, model.getSecondaryModel());
            ospSetData(_renderer, RENDERER_PROPERTY_USER_DATA, model.simulationData());
            ospSetObject(_renderer, DEFAULT_COMMON_TRANSFER_FUNCTION, model.transferFunction());
        }
        else
        {
            // ospRemoveParam leaks objects, so we set it to null first
            ospSetData(_renderer, RENDERER_PROPERTY_USER_DATA, nullptr);
            ospRemoveParam(_renderer, RENDERER_PROPERTY_USER_DATA);
        }

        // Setting the clip planes in the renderer and the camera
        Planes planes;
        for (const auto& clipPlane : scene.getClipPlanes())
            planes.push_back(clipPlane->getPlane());

        setClipPlanes(planes);
        _camera->setClipPlanes(planes);
    }
    // Anaglyph property is handled by the camera, but processed by the renderer
    Vector3f anaglyphIpdOffset;
    bool anaglyphEnabled = false;
    if (_camera->getCurrentType() == CAMERA_PROPERTY_TYPE_ANAGLYPH)
    {
        const Vector3d dir = _camera->getOrientation() * UP_VECTOR;
        const double d = dot(dir, UP_VECTOR);
        const Vector3d dir_du = (d > 0.999 ? Vector3d(1, 0, 0) : normalize(cross(dir, UP_VECTOR))); // Avoid gimble lock
        const double interpupillaryDistance =
            _camera->getProperty<double>(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str());
        anaglyphIpdOffset = 0.5f * interpupillaryDistance * dir_du;
        anaglyphEnabled = true;
    }

    osphelper::set(_renderer, OSPRAY_RENDERER_PROPERTY_ANAGLYPH_IPD_OFFSET, anaglyphIpdOffset);
    osphelper::set(_renderer, OSPRAY_RENDERER_PROPERTY_ANAGLYPH_ENABLED, static_cast<int>(anaglyphEnabled));

    _camera->commit();

    osphelper::set(_renderer, RENDERER_PROPERTY_TIMESTAMP.name.c_str(), static_cast<float>(ap.getFrame()));
    osphelper::set(_renderer, RENDERER_PROPERTY_RANDOM_NUMBER, rand() % 10000);
    osphelper::set(_renderer, OSPRAY_RENDERER_PROPERTY_VARIANCE_THRESHOLD,
                   static_cast<float>(rp.getVarianceThreshold()));
    osphelper::set(_renderer, OSPRAY_RENDERER_PROPERTY_SAMPLES_PER_PIXEL, static_cast<int>(_spp));

    if (auto material = std::static_pointer_cast<OSPRayMaterial>(scene.getBackgroundMaterial()))
    {
        material->setDiffuseColor(_backgroundColor);
        material->commit(_currentOSPRenderer);
        ospSetObject(_renderer, RENDERER_PROPERTY_BACKGROUND_MATERIAL, material->getOSPMaterial());
    }

    ospSetObject(_renderer, OSPRAY_RENDERER_PROPERTY_CAMERA, _camera->impl());
    ospSetObject(_renderer, OSPRAY_RENDERER_PROPERTY_WORLD, scene.getModel());

    // Clip planes
    if (!_clipPlanes.empty())
    {
        const auto clipPlanes = convertVectorToFloat(_clipPlanes);
        auto clipPlaneData = ospNewData(clipPlanes.size(), OSP_FLOAT4, clipPlanes.data());
        ospSetData(_renderer, CAMERA_PROPERTY_CLIPPING_PLANES, clipPlaneData);
        ospRelease(clipPlaneData);
    }
    else
    {
        // ospRemoveParam leaks objects, so we set it to null first
        ospSetData(_renderer, CAMERA_PROPERTY_CLIPPING_PLANES, nullptr);
        ospRemoveParam(_renderer, CAMERA_PROPERTY_CLIPPING_PLANES);
    }

    ospCommit(_renderer);
}

void OSPRayRenderer::setCamera(CameraPtr camera)
{
    _camera = static_cast<OSPRayCamera*>(camera.get());
    assert(_camera);
    if (_renderer)
        ospSetObject(_renderer, OSPRAY_RENDERER_PROPERTY_CAMERA, _camera->impl());
    markModified();
}

Renderer::PickResult OSPRayRenderer::pick(const Vector2f& pickPos)
{
    OSPPickResult ospResult;
    osp::vec2f pos{pickPos.x, pickPos.y};

    // HACK: as the time for picking is set to 0.5 and interpolated in a (default) 0..0 range, the ray.time will be 0.
    // So all geometries that have a time > 0 (like branches that have distance to the soma for the growing use-case),
    // cannot be picked. So we make the range as large as possible to make ray.time be as large as possible.
    osphelper::set(_camera->impl(), OSPRAY_RENDERER_PROPERTY_SHUTTER_CLOSE, INFINITY);
    ospCommit(_camera->impl());

    ospPick(&ospResult, _renderer, pos);

    // UNDO HACK
    osphelper::set(_camera->impl(), OSPRAY_RENDERER_PROPERTY_SHUTTER_CLOSE, 0.f);
    ospCommit(_camera->impl());

    PickResult result;
    result.hit = ospResult.hit;
    if (result.hit)
    {
        result.pos = {ospResult.position.x, ospResult.position.y, ospResult.position.z};
        CORE_INFO("OSPRAY Depth: " << pickPos << " - " << result.pos);
    }
    return result;
}

void OSPRayRenderer::_createOSPRenderer()
{
    auto newRenderer = ospNewRenderer(getCurrentType().c_str());
    if (!newRenderer)
        throw std::runtime_error(getCurrentType() + " is not a registered renderer");
    _destroyRenderer();
    _renderer = newRenderer;
    if (_camera)
        ospSetObject(_renderer, OSPRAY_RENDERER_PROPERTY_CAMERA, _camera->impl());
    _currentOSPRenderer = getCurrentType();
    markModified(false);
}

void OSPRayRenderer::_commitRendererMaterials()
{
    OSPRayScene& scene = static_cast<OSPRayScene&>(_engine->getScene());
    scene.visitModels([&renderer = _currentOSPRenderer](Model& model)
                      { static_cast<OSPRayModel&>(model).commitMaterials(renderer); });
}

void OSPRayRenderer::setClipPlanes(const Planes& planes)
{
    if (_clipPlanes == planes)
        return;
    _clipPlanes = planes;
    markModified(false);
}

} // namespace ospray
} // namespace engine
} // namespace core