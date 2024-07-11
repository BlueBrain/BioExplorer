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

#include "OSPRayScene.h"
#include "OSPRayMaterial.h"
#include "OSPRayModel.h"
#include "OSPRayProperties.h"
#include "OSPRayUtils.h"
#include "OSPRayVolume.h"

#include <platform/core/common/ImageManager.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/Transformation.h>
#include <platform/core/common/light/Light.h>
#include <platform/core/engineapi/Model.h>

#include <platform/core/parameters/FieldParameters.h>
#include <platform/core/parameters/GeometryParameters.h>
#include <platform/core/parameters/VolumeParameters.h>

namespace core
{
namespace engine
{
namespace ospray
{
OSPRayScene::OSPRayScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                         VolumeParameters& volumeParameters, FieldParameters& fieldParameters)
    : Scene(animationParameters, geometryParameters, volumeParameters, fieldParameters)
    , _memoryManagementFlags(geometryParameters.getMemoryMode() == MemoryMode::shared ? uint32_t(OSP_DATA_SHARED_BUFFER)
                                                                                      : 0)
{
    _backgroundMaterial = std::make_shared<OSPRayMaterial>(PropertyMap(), true);
}

OSPRayScene::~OSPRayScene()
{
    _destroyLights();
    if (_rootModel)
        ospRelease(_rootModel);
}
void OSPRayScene::_destroyLights()
{
    for (auto& light : _ospLights)
        ospRelease(light);
    _ospLights.clear();

    ospRelease(_ospLightData);
    _ospLightData = nullptr;
}

void OSPRayScene::commit()
{
    Scene::commit();
    commitLights();

    // copy the list to avoid locking the mutex
    ModelDescriptors modelDescriptors;
    {
        auto lock = acquireReadAccess();
        modelDescriptors = _modelDescriptors;
    }

    const bool rebuildScene = isModified();
    const bool addRemoveVolumes = _commitVolumeAndTransferFunction(modelDescriptors);

    if (!rebuildScene && !addRemoveVolumes)
    {
        // check for dirty models aka their geometry has been altered
        bool doUpdate = false;
        for (auto& modelDescriptor : modelDescriptors)
        {
            auto& model = modelDescriptor->getModel();
            if (_fieldParameters.isModified())
            {
                auto& impl = static_cast<OSPRayModel&>(model);
                impl.commitFieldParameters();
            }

            if (model.isDirty())
            {
                model.commitGeometry();
                // need to continue re-adding the models to update the bounding
                // box model to reflect the new model size
                doUpdate = true;
            }
        }
        if (!doUpdate)
            return;
    }

    _activeModels.clear();

    if (_rootModel)
        ospRelease(_rootModel);
    _rootModel = ospNewModel();

    for (auto modelDescriptor : modelDescriptors)
    {
        if (!modelDescriptor->getEnabled())
            continue;

        // keep models from being deleted via removeModel() as long as we use
        // them here
        _activeModels.push_back(modelDescriptor);

        auto& impl = static_cast<OSPRayModel&>(modelDescriptor->getModel());
        const auto& transformation = modelDescriptor->getTransformation();

        CORE_DEBUG("Committing " << modelDescriptor->getName());

        impl.commitGeometry();
        impl.logInformation();

        // add volumes to root model, because scivis renderer does not consider
        // volumes from instances
        if (modelDescriptor->getVisible())
            for (auto volume : modelDescriptor->getModel().getVolumes())
            {
                auto ospVolume = std::dynamic_pointer_cast<OSPRayVolume>(volume.second);
                ospAddVolume(_rootModel, ospVolume->impl());
            }

        const auto& instances = modelDescriptor->getInstances();
        for (size_t i = 0; i < instances.size(); ++i)
        {
            const auto& instance = instances[i];

            // First instance uses model transformation
            const auto& instanceTransform = (i == 0 ? transformation : instance.getTransformation());

            if (modelDescriptor->getBoundingBox() && instance.getBoundingBox())
            {
                // scale and move the unit-sized bounding box geometry to the
                // model size/scale first, then apply the instance transform
                const auto& modelBounds = modelDescriptor->getModel().getBounds();
                Transformation modelTransform;
                modelTransform.setTranslation(modelBounds.getCenter() - 0.5 * modelBounds.getSize());
                modelTransform.setScale(modelBounds.getSize());

                addInstance(_rootModel, impl.getBoundingBoxModel(),
                            transformationToAffine3f(instanceTransform) * transformationToAffine3f(modelTransform));
            }

            if (modelDescriptor->getVisible() && instance.getVisible())
                addInstance(_rootModel, impl.getPrimaryModel(), instanceTransform);
        }

        impl.markInstancesClean();
    }
    CORE_DEBUG("Committing root models");

    ospCommit(_rootModel);

    computeBounds();
}

bool OSPRayScene::commitLights()
{
    if (!_lightManager.isModified())
        return false;

    _destroyLights();

    for (const auto& kv : _lightManager.getLights())
    {
        auto baseLight = kv.second;
        OSPLight ospLight{nullptr};

        switch (baseLight->_type)
        {
        case LightType::DIRECTIONAL:
        {
            ospLight = ospNewLight3(OSPRAY_LIGHT_PROPERTY_DISTANT);
            const auto light = static_cast<DirectionalLight*>(baseLight.get());
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_DIRECTION, Vector3f(light->_direction));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_DISTANT_ANGULAR_DIAMETER,
                           static_cast<float>(light->_angularDiameter));
            break;
        }
        case LightType::SPHERE:
        {
            ospLight = ospNewLight3(OSPRAY_LIGHT_PROPERTY_POINT);
            const auto light = static_cast<SphereLight*>(baseLight.get());
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_POSITION, Vector3f(light->_position));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_RADIUS, static_cast<float>(light->_radius));
            break;
        }
        case LightType::QUAD:
        {
            ospLight = ospNewLight3(OSPRAY_LIGHT_PROPERTY_QUAD);
            const auto light = static_cast<QuadLight*>(baseLight.get());
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_POSITION, Vector3f(light->_position));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_QUAD_EDGE1, Vector3f(light->_edge1));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_QUAD_EDGE2, Vector3f(light->_edge2));
            break;
        }
        case LightType::SPOTLIGHT:
        {
            ospLight = ospNewLight3(OSPRAY_LIGHT_PROPERTY_SPOT);
            const auto light = static_cast<SpotLight*>(baseLight.get());
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_POSITION, Vector3f(light->_position));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_DIRECTION, Vector3f(light->_direction));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_SPOT_OPENING_ANGLE,
                           static_cast<float>(light->_openingAngle));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_SPOT_PENUMBRA_ANGLE,
                           static_cast<float>(light->_penumbraAngle));
            osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_RADIUS, static_cast<float>(light->_radius));
            break;
        }
        case LightType::AMBIENT:
        {
            ospLight = ospNewLight3(OSPRAY_LIGHT_PROPERTY_AMBIENT);
            break;
        }
        }

        assert(ospLight);

        osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_COLOR, Vector3f(baseLight->_color));
        osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_INTENSITY, static_cast<float>(baseLight->_intensity));
        osphelper::set(ospLight, OSPRAY_LIGHT_PROPERTY_IS_VISIBLE, baseLight->_isVisible);

        _ospLights.push_back(ospLight);
        ospCommit(ospLight);
    }

    // NOTE: since the lights are shared between scene and renderer we let
    // OSPRay allocate a new buffer to avoid use-after-free issues
    const size_t memoryFlags = 0;
    _ospLightData = ospNewData(_ospLights.size(), OSP_OBJECT, _ospLights.data(), memoryFlags);
    ospCommit(_ospLightData);

    return true;
}

bool OSPRayScene::_commitVolumeAndTransferFunction(ModelDescriptors& modelDescriptors)
{
    bool rebuildScene = false;
    for (auto& modelDescriptor : modelDescriptors)
    {
        auto& model = static_cast<OSPRayModel&>(modelDescriptor->getModel());
        const bool dirtyTransferFunction = model.commitTransferFunction();
        const bool dirtySimulationData = model.commitSimulationData();

        if (dirtyTransferFunction || dirtySimulationData)
            markModified(false);
        if (model.isVolumesDirty())
        {
            rebuildScene = true;
            model.resetVolumesDirty();
        }
        for (auto& volume : model.getVolumes())
        {
            if (volume.second->isModified() || rebuildScene || _volumeParameters.isModified())
            {
                volume.second->commit();
                // to reset accumulation if new blocks are added
                markModified(false);
            }
        }
    }
    return rebuildScene;
}

ModelPtr OSPRayScene::createModel() const
{
    return std::make_unique<OSPRayModel>(_animationParameters, _volumeParameters, _geometryParameters,
                                         _fieldParameters);
}

ModelDescriptorPtr OSPRayScene::getSimulatedModel()
{
    auto lock = acquireReadAccess();
    for (auto model : _modelDescriptors)
    {
        const auto& ospModel = static_cast<const OSPRayModel&>(model->getModel());
        if (ospModel.simulationData())
            return model;
    }
    return ModelDescriptorPtr{};
}
} // namespace ospray
} // namespace engine
} // namespace core