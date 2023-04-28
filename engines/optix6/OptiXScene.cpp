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

#include "OptiXScene.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXModel.h"
#include "OptiXVolume.h"

#include <brayns/common/light/Light.h>
#include <brayns/common/log.h>
#include <brayns/common/utils/utils.h>
#include <brayns/engineapi/Material.h>
#include <brayns/parameters/ParametersManager.h>

#include <optixu/optixu_math_stream_namespace.h>

namespace brayns
{
OptiXScene::OptiXScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                       VolumeParameters& volumeParameters)
    : Scene(animationParameters, geometryParameters, volumeParameters)
    , _lightBuffer(nullptr)
{
    _backgroundMaterial = std::make_shared<OptiXMaterial>();
    auto oc = OptiXContext::get().getOptixContext();

    // To avoid crashes we need to initialize some buffers and variables
    // even if they are not always used in CUDA kernels.

    { // Create dummy texture sampler
        ::optix::TextureSampler sampler = oc->createTextureSampler();
        sampler->setArraySize(1u);
        optix::Buffer buffer = oc->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1, 1);
        sampler->setBuffer(buffer);
        _dummyTextureSampler = sampler;
    }

    // Create dummy simulation data
    oc["simulation_data"]->setBuffer(oc->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 0));
}

OptiXScene::~OptiXScene() = default;

bool OptiXScene::commitLights()
{
    if (!_lightManager.isModified())
        return false;

    if (_lightManager.getLights().empty())
    {
        BRAYNS_ERROR("No lights are currently defined");
        return false;
    }

    _optixLights.clear();

    for (const auto& kv : _lightManager.getLights())
    {
        auto baseLight = kv.second;

        switch (baseLight->_type)
        {
        case LightType::SPHERE:
        {
            const auto light = static_cast<SphereLight*>(baseLight.get());
            const Vector3f position = light->_position;
            const Vector3f color = light->_color;
            BasicLight optixLight = {{position.x, position.y, position.z},
                                     {color.x, color.y, color.z},
                                     1, // Casts shadows
                                     BASIC_LIGHT_TYPE_POINT};
            _optixLights.push_back(optixLight);
            break;
        }
        case LightType::DIRECTIONAL:
        {
            const auto light = static_cast<DirectionalLight*>(baseLight.get());
            const Vector3f direction = light->_direction;
            const Vector3f color = light->_color;
            BasicLight optixLight = {{direction.x, direction.y, direction.z},
                                     {color.x, color.y, color.z},
                                     1, // Casts shadows
                                     BASIC_LIGHT_TYPE_DIRECTIONAL};
            _optixLights.push_back(optixLight);
            break;
        }
        default:
        {
            BRAYNS_WARN("Unsupported light type");
            break;
        }
        }
    }

    if (_lightBuffer)
        _lightBuffer->destroy();

    auto context = OptiXContext::get().getOptixContext();
    _lightBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, _optixLights.size());
    _lightBuffer->setElementSize(sizeof(BasicLight));
    memcpy(_lightBuffer->map(), _optixLights.data(), _optixLights.size() * sizeof(_optixLights[0]));
    _lightBuffer->unmap();
    context[CONTEXT_LIGHTS]->set(_lightBuffer);

    return true;
}

ModelPtr OptiXScene::createModel() const
{
    return std::make_unique<OptiXModel>(_animationParameters, _volumeParameters);
}

void OptiXScene::_commitVolumeParameters()
{
    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_VOLUME_GRADIENT_SHADING_ENABLED]->setUint(_volumeParameters.getGradientShading());
    context[CONTEXT_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE]->setFloat(_volumeParameters.getAdaptiveMaxSamplingRate());
    context[CONTEXT_VOLUME_ADAPTIVE_SAMPLING]->setUint(_volumeParameters.getAdaptiveSampling());
    context[CONTEXT_VOLUME_SINGLE_SHADE]->setUint(_volumeParameters.getSingleShade());
    context[CONTEXT_VOLUME_PRE_INTEGRATION]->setUint(_volumeParameters.getPreIntegration());
    context[CONTEXT_VOLUME_SAMPLING_RATE]->setFloat(_volumeParameters.getSamplingRate());
    const Vector3f specular = _volumeParameters.getSpecular();
    context[CONTEXT_VOLUME_SPECULAR_COLOR]->setFloat(specular.x, specular.y, specular.z);
    const auto boxLower = _volumeParameters.getClipBox().getMin();
    context[CONTEXT_VOLUME_CLIPPING_BOX_LOWER]->setFloat(boxLower.x, boxLower.y, boxLower.z);
    const auto boxUpper = _volumeParameters.getClipBox().getMin();
    context[CONTEXT_VOLUME_CLIPPING_BOX_UPPER]->setFloat(boxUpper.x, boxUpper.y, boxUpper.z);
}

void OptiXScene::commit()
{
    // Always upload transfer function and simulation data if changed
    for (size_t i = 0; i < _modelDescriptors.size(); ++i)
    {
        auto& model = _modelDescriptors[i]->getModel();
        model.commitSimulationData();

        _commitVolumeParameters();
        if (model.commitTransferFunction())
            markModified();
    }

    commitLights();

    if (!isModified())
        return;

    // Remove all models marked for removal
    for (auto& model : _modelDescriptors)
        if (model->isMarkedForRemoval())
            model->callOnRemoved();

    _modelDescriptors.erase(std::remove_if(_modelDescriptors.begin(), _modelDescriptors.end(),
                                           [](const auto& m) { return m->isMarkedForRemoval(); }),
                            _modelDescriptors.end());

    auto context = OptiXContext::get().getOptixContext();

    auto values = std::map<TextureType, std::string>{{TextureType::diffuse, "envmap"},
                                                     {TextureType::radiance, "envmap_radiance"},
                                                     {TextureType::irradiance, "envmap_irradiance"},
                                                     {TextureType::brdf_lut, "envmap_brdf_lut"}};
    if (hasEnvironmentMap())
        _backgroundMaterial->commit();

    auto optixMat = std::static_pointer_cast<OptiXMaterial>(_backgroundMaterial);
    for (const auto& i : values)
    {
        auto sampler = _dummyTextureSampler;
        if (hasEnvironmentMap() && optixMat->hasTexture(i.first))
            sampler = optixMat->getTextureSampler(i.first);
        context[i.second]->setInt(sampler->getId());
        if (i.first == TextureType::radiance && _backgroundMaterial->hasTexture(TextureType::radiance))
        {
            const auto& radianceTex = _backgroundMaterial->getTexture(TextureType::radiance);
            context[CONTEXT_MATERIAL_RADIANCE_LODS]->setUint(radianceTex->getMipLevels() - 1);
        }
    }

    context[CONTEXT_USE_ENVIRONMENT_MAP]->setUint(hasEnvironmentMap() ? 1 : 0);

    // Geometry
    if (_rootGroup)
        _rootGroup->destroy();

    _rootGroup = OptiXContext::get().createGroup();

    for (size_t i = 0; i < _modelDescriptors.size(); ++i)
    {
        auto& modelDescriptor = _modelDescriptors[i];
        if (!modelDescriptor->getEnabled())
            continue;

        auto& impl = static_cast<OptiXModel&>(modelDescriptor->getModel());

        BRAYNS_DEBUG("Committing " << modelDescriptor->getName());

        impl.commitGeometry();
        impl.logInformation();

        if (modelDescriptor->getVisible())
        {
            const auto geometryGroup = impl.getGeometryGroup();
            const auto& instances = modelDescriptor->getInstances();
            size_t count{0};
            for (const auto& instance : instances)
            {
                auto transformation = instance.getTransformation();
                if (count == 0)
                    transformation = modelDescriptor->getTransformation();
                const ::glm::mat4 matrix = transformation.toMatrix(true);
                ::optix::Matrix4x4 optixMatrix(glm::value_ptr(matrix));
                ::optix::Transform xform = context->createTransform();
                xform->setChild(geometryGroup);
                xform->setMatrix(true, optixMatrix.getData(), optixMatrix.inverse().getData());
                _rootGroup->addChild(xform);
                ++count;
            }
            BRAYNS_DEBUG("Group has " << geometryGroup->getChildCount() << " children");
        }

        if (modelDescriptor->getBoundingBox())
        {
            // scale and move the unit-sized bounding box geometry to the
            // model size/scale first, then apply the instance transform
            const auto boundingBoxGroup = impl.getBoundingBoxGroup();
            ::optix::Transform xform = context->createTransform();

            const auto& modelBounds = modelDescriptor->getModel().getBounds();
            Transformation modelTransform;
            modelTransform.setTranslation(modelBounds.getCenter() / modelBounds.getSize() - Vector3d(0.5));
            modelTransform.setScale(modelBounds.getSize());

            Matrix4f mtxd = modelTransform.toMatrix(true);
            mtxd = glm::transpose(mtxd);
            auto trf = glm::value_ptr(mtxd);

            xform->setMatrix(false, trf, 0);
            xform->setChild(boundingBoxGroup);
            _rootGroup->addChild(xform);
        }
    }
    computeBounds();

    BRAYNS_DEBUG("Root has " << _rootGroup->getChildCount() << " children");

    context[CONTEXT_SCENE_TOP_OBJECT]->set(_rootGroup);
    context[CONTEXT_SCENE_TOP_SHADOWER]->set(_rootGroup);

    // TODO: triggers the change callback to re-broadcast the scene if the clip
    // planes have changed. Provide an RPC to update/set clip planes.
    markModified();
}

} // namespace brayns
