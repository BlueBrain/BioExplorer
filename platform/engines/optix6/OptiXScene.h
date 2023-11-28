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
               VolumeParameters& volumeParameters);
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