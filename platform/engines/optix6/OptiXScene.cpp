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

#include "OptiXScene.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXModel.h"
#include "OptiXUtils.h"
#include "OptiXVolume.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/light/Light.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/parameters/ParametersManager.h>

#include <optixu/optixu_math_stream_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
OptiXScene::OptiXScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                       VolumeParameters& volumeParameters)
    : Scene(animationParameters, geometryParameters, volumeParameters)
    , _lightBuffer(nullptr)
{
    _backgroundMaterial = std::make_shared<OptiXMaterial>();
    auto context = OptiXContext::get().getOptixContext();

    // To avoid crashes we need to initialize some buffers and variables
    // even if they are not always used in CUDA kernels.
    { // Create dummy texture sampler
        ::optix::TextureSampler sampler = context->createTextureSampler();
        sampler->setArraySize(1u);
        ::optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1, 1);
        sampler->setBuffer(buffer);
        _dummyTextureSampler = sampler;
    }

    // Create dummy simulation data
    context[CONTEXT_USER_DATA]->setBuffer(context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 0));
}

OptiXScene::~OptiXScene()
{
    RT_DESTROY(_lightBuffer);
    RT_DESTROY(_colorMapBuffer);
    RT_DESTROY(_emissionIntensityMapBuffer);
    RT_DESTROY(_backgroundTextureSampler);
    RT_DESTROY(_dummyTextureSampler);
    RT_DESTROY(_volumeBuffer);
    RT_DESTROY(_clipPlanesBuffer);
}

bool OptiXScene::commitLights()
{
    if (!_lightManager.isModified())
        return false;

    if (_lightManager.getLights().empty())
    {
        CORE_ERROR("No lights are currently defined");
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
                                     1, // Casts shadowIntensity
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
                                     1, // Casts shadowIntensity
                                     BASIC_LIGHT_TYPE_DIRECTIONAL};
            _optixLights.push_back(optixLight);
            break;
        }
        default:
        {
            CORE_WARN("Unsupported light type");
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
    return std::make_unique<OptiXModel>(_animationParameters, _volumeParameters, _geometryParameters);
}

void OptiXScene::_commitVolumeParameters()
{
    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_VOLUME_GRADIENT_SHADING_ENABLED]->setUint(_volumeParameters.getGradientShading());
    context[CONTEXT_VOLUME_GRADIENT_OFFSET]->setFloat(_volumeParameters.getGradientOffset());
    context[CONTEXT_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE]->setFloat(_volumeParameters.getAdaptiveMaxSamplingRate());
    context[CONTEXT_VOLUME_ADAPTIVE_SAMPLING]->setUint(_volumeParameters.getAdaptiveSampling());
    context[CONTEXT_VOLUME_SINGLE_SHADE]->setUint(_volumeParameters.getSingleShade());
    context[CONTEXT_VOLUME_PRE_INTEGRATION]->setUint(_volumeParameters.getPreIntegration());
    context[CONTEXT_VOLUME_SAMPLING_RATE]->setFloat(_volumeParameters.getSamplingRate());
    const Vector3f specular = _volumeParameters.getSpecular();
    context[CONTEXT_VOLUME_SPECULAR_COLOR]->setFloat(specular.x, specular.y, specular.z);
    const Vector3f userParameters = _volumeParameters.getUserParameters();
    context[CONTEXT_VOLUME_USER_PARAMETERS]->setFloat(userParameters.x, userParameters.y, userParameters.z);
    const auto boxLower = _volumeParameters.getClipBox().getMin();
    context[CONTEXT_VOLUME_CLIPPING_BOX_LOWER]->setFloat(boxLower.x, boxLower.y, boxLower.z);
    const auto boxUpper = _volumeParameters.getClipBox().getMin();
    context[CONTEXT_VOLUME_CLIPPING_BOX_UPPER]->setFloat(boxUpper.x, boxUpper.y, boxUpper.z);
}

void OptiXScene::_commitGeometryParameters()
{
    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_GEOMETRY_SDF_EPSILON]->setFloat(_geometryParameters.getSdfEpsilon());
    context[CONTEXT_GEOMETRY_SDF_NB_MARCH_ITERATIONS]->setUint(_geometryParameters.getSdfNbMarchIterations());
    context[CONTEXT_GEOMETRY_SDF_BLEND_FACTOR]->setFloat(_geometryParameters.getSdfBlendFactor());
    context[CONTEXT_GEOMETRY_SDF_BLEND_LERP_FACTOR]->setFloat(_geometryParameters.getSdfBlendLerpFactor());
    context[CONTEXT_GEOMETRY_SDF_OMEGA]->setFloat(_geometryParameters.getSdfOmega());
    context[CONTEXT_GEOMETRY_SDF_DISTANCE]->setFloat(_geometryParameters.getSdfDistance());
}

void OptiXScene::_commitClippingPlanes()
{
    if (!isModified())
        return;

    auto context = OptiXContext::get().getOptixContext();

    const size_t numClipPlanes = _clipPlanes.size();
    const auto size = numClipPlanes * sizeof(Vector4f);

    RT_DESTROY(_clipPlanesBuffer);
    _clipPlanesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numClipPlanes);

    if (numClipPlanes > 0)
    {
        Vector4fs buffer;
        buffer.reserve(numClipPlanes);
        for (const auto clipPlane : _clipPlanes)
        {
            const auto& p = clipPlane->getPlane();
            buffer.push_back({static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
                              static_cast<float>(p[3])});
        }

        memcpy(_clipPlanesBuffer->map(), buffer.data(), size);
        _clipPlanesBuffer->unmap();
    }

    context[CONTEXT_CLIPPING_PLANES]->setBuffer(_clipPlanesBuffer);
    context[CONTEXT_NB_CLIPPING_PLANES]->setUint(numClipPlanes);
}

void OptiXScene::commit()
{
    // Always upload transfer function and simulation data if changed
    for (size_t i = 0; i < _modelDescriptors.size(); ++i)
    {
        auto& model = _modelDescriptors[i]->getModel();
        model.commitSimulationData();
        _commitVolumeParameters();
    }

    commitLights();
    _commitGeometryParameters();
    _commitClippingPlanes();

    if (isModified())
    {
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

            CORE_DEBUG("Committing " << modelDescriptor->getName());

            impl.commitGeometry();
            impl.logInformation();

            if (modelDescriptor->getVisible())
            {
                const auto geometryGroup = impl.getGeometryGroup();
                const auto& instances = modelDescriptor->getInstances();
                size_t count{0};
                for (const auto& instance : instances)
                {
                    auto modelTransformation = instance.getTransformation();
                    if (count == 0)
                        modelTransformation = modelDescriptor->getTransformation();
                    const ::glm::mat4 matrix = modelTransformation.toMatrix(true);
                    ::optix::Matrix4x4 optixMatrix(glm::value_ptr(matrix));
                    ::optix::Transform instanceTransformation = context->createTransform();
                    instanceTransformation->setChild(geometryGroup);
                    instanceTransformation->setMatrix(true, optixMatrix.getData(), optixMatrix.inverse().getData());
                    _rootGroup->addChild(instanceTransformation);
                    ++count;
                }
                CORE_DEBUG("Group has " << geometryGroup->getChildCount() << " children");
            }

            if (modelDescriptor->getBoundingBox())
            {
                // scale and move the unit-sized bounding box geometry to the model size/scale first, then apply the
                // instance transform
                const auto boundingBoxGroup = impl.getBoundingBoxGroup();
                ::optix::Transform transformation = context->createTransform();

                const auto& modelBounds = modelDescriptor->getModel().getBounds();
                Transformation modelTransformation;
                modelTransformation.setTranslation(modelBounds.getCenter() / modelBounds.getSize() - Vector3d(0.5));
                modelTransformation.setScale(modelBounds.getSize());

                Matrix4f modelMatrix = modelTransformation.toMatrix(true);
                modelMatrix = glm::transpose(modelMatrix);
                const auto trf = glm::value_ptr(modelMatrix);

                transformation->setMatrix(false, trf, 0);
                transformation->setChild(boundingBoxGroup);
                _rootGroup->addChild(transformation);
            }
        }
        computeBounds();

        CORE_DEBUG("Root has " << _rootGroup->getChildCount() << " children");

        context[CONTEXT_SCENE_TOP_OBJECT]->set(_rootGroup);
        context[CONTEXT_SCENE_TOP_SHADOWER]->set(_rootGroup);

        // TODO: triggers the change callback to re-broadcast the scene if the clip planes have changed. Provide an RPC
        // to update/set clip planes.
        markModified();
    }

    for (size_t i = 0; i < _modelDescriptors.size(); ++i)
    {
        auto& model = _modelDescriptors[i]->getModel();
        if (model.commitTransferFunction())
            markModified();
    }
}
} // namespace optix
} // namespace engine
} // namespace core