/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#ifndef OSPRAYSCENE_H
#define OSPRAYSCENE_H

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

#include <ospray.h>

namespace core
{
/**

   OSPRay specific scene

   This object is the OSPRay specific implementation of a scene

*/
class OSPRayScene : public Scene
{
public:
    OSPRayScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                VolumeParameters& volumeParameters);
    ~OSPRayScene();

    /** @copydoc Scene::commit */
    void commit() final;

    /** @copydoc Scene::commitLights */
    bool commitLights() final;

    /** @copydoc Scene::supportsConcurrentSceneUpdates. */
    bool supportsConcurrentSceneUpdates() const final { return true; }
    ModelPtr createModel() const final;

    OSPModel getModel() { return _rootModel; }
    OSPData lightData() { return _ospLightData; }
    ModelDescriptorPtr getSimulatedModel();

private:
    bool _commitVolumeAndTransferFunction(ModelDescriptors& modelDescriptors);
    void _destroyLights();

    OSPModel _rootModel{nullptr};

    std::vector<OSPLight> _ospLights;

    OSPData _ospLightData{nullptr};

    size_t _memoryManagementFlags{0};

    ModelDescriptors _activeModels;
};
} // namespace core
#endif // OSPRAYSCENE_H
