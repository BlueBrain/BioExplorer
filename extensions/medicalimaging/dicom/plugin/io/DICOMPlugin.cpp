/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "DICOMPlugin.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Properties.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/Properties.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <DICOM_generated_DICOM.cu.ptx.h>
#include <platform/engines/optix6/OptiXCommonStructs.h>
#include <platform/engines/optix6/OptiXContext.h>
#include <platform/engines/optix6/OptiXProperties.h>
#endif

namespace medicalimagingexplorer
{
namespace dicom
{
#define REGISTER_LOADER(LOADER, FUNC) registry.registerLoader({std::bind(&LOADER::getSupportedDataTypes), FUNC});

const std::string RENDERER_DICOM = "dicom";

using namespace core;

void _addDICOMRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_DICOM);
    core::PropertyMap properties;
    properties.setProperty(RENDERER_PROPERTY_FOG_START);
    properties.setProperty(RENDERER_PROPERTY_FOG_THICKNESS);
    properties.setProperty(RENDERER_PROPERTY_SHADOW_INTENSITY);
    properties.setProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH);
    properties.setProperty(COMMON_PROPERTY_EXPOSURE);
    properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH);
    properties.setProperty(DICOM_RENDERER_PROPERTY_SURFACE_OFFSET);
    properties.setProperty({RENDERER_PROPERTY_MAX_RAY_DEPTH.name, DEFAULT_RENDERER_RAY_DEPTH, 1,
                            static_cast<int>(DEFAULT_RENDERER_MAX_RAY_DEPTH),
                            RENDERER_PROPERTY_MAX_RAY_DEPTH.metaData});
    engine.addRendererType(RENDERER_DICOM, properties);
}

DICOMPlugin::DICOMPlugin(PropertyMap &&dicomParams)
    : ExtensionPlugin()
    , _dicomParams(dicomParams)
{
}

void DICOMPlugin::init()
{
    auto &scene = _api->getScene();
    auto &registry = scene.getLoaderRegistry();
    registry.registerLoader(
        std::make_unique<DICOMLoader>(scene, std::move(_api->getParametersManager().getGeometryParameters()),
                                      std::move(_dicomParams)));

    auto &engine = _api->getEngine();
    auto &params = engine.getParametersManager().getApplicationParameters();
    auto &engineName = params.getEngine();
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        _createOptiXRenderers();
        _createRenderers();
    }
#endif
    engine.setRendererType(core::RENDERER_PROPERTY_TYPE_ADVANCED);
}

#ifdef USE_OPTIX6
void DICOMPlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_DICOM, DICOM_generated_DICOM_cu_ptx},
    };
    core::engine::optix::OptiXContext &context = core::engine::optix::OptiXContext::get();
    for (const auto &renderer : renderers)
    {
        const std::string ptx = renderer.second;

        auto osp = std::make_shared<core::engine::optix::OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, core::engine::optix::OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE);
        osp->closest_hit_textured = context.getOptixContext()->createProgramFromPTXString(
            ptx, core::engine::optix::OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE_TEXTURED);
        osp->any_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, core::engine::optix::OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW);
        osp->exception_program =
            context.getOptixContext()->createProgramFromPTXString(ptx,
                                                                  core::engine::optix::OPTIX_CUDA_FUNCTION_EXCEPTION);
        context.addRenderer(renderer.first, osp);
    }
}
#endif

void DICOMPlugin::_createRenderers()
{
    auto &engine = _api->getEngine();
    _addDICOMRenderer(engine);
}

extern "C" ExtensionPlugin *core_plugin_create(int /*argc*/, char ** /*argv*/)
{
    PLUGIN_INFO("");
    PLUGIN_INFO(" _|_|_|    _|_|_|    _|_|_|    _|_|    _|      _|");
    PLUGIN_INFO(" _|    _|    _|    _|        _|    _|  _|_|  _|_|");
    PLUGIN_INFO(" _|    _|    _|    _|        _|    _|  _|  _|  _|");
    PLUGIN_INFO(" _|    _|    _|    _|        _|    _|  _|      _|");
    PLUGIN_INFO(" _|_|_|    _|_|_|    _|_|_|    _|_|    _|      _|");
    PLUGIN_INFO("");

    return new DICOMPlugin(DICOMLoader::getCLIProperties());
}
} // namespace dicom
} // namespace medicalimagingexplorer