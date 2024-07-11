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

#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Scene.h>

#include <ospray.h>

namespace core
{
namespace engine
{
namespace ospray
{
/**

   OSPRay specific scene

   This object is the OSPRay specific implementation of a scene

*/
class OSPRayScene : public Scene
{
public:
    OSPRayScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                VolumeParameters& volumeParameters, FieldParameters& fieldParameters);
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
} // namespace ospray
} // namespace engine
} // namespace core