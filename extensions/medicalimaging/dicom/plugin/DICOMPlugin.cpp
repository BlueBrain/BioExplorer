/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "DICOMPlugin.h"

#include <plugin/common/Logs.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <DICOM_generated_Volume.cu.ptx.h>
#include <platform/engines/optix6/OptiXContext.h>
#endif

namespace medicalimagingexplorer
{
namespace dicom
{
#define REGISTER_LOADER(LOADER, FUNC) registry.registerLoader({std::bind(&LOADER::getSupportedDataTypes), FUNC});

const std::string RENDERER_VOLUME = "volume";

using namespace core;

void _addVolumeRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_VOLUME);
    core::PropertyMap properties;
    properties.setProperty({"shadows", 0., 0., 1., {"Shadows"}});
    properties.setProperty({"softShadows", 0., 0., 1., {"Soft shadows"}});
    engine.addRendererType(RENDERER_VOLUME, properties);
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
    const auto &engineName = params.getEngine();
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        _createOptiXRenderers();
        _createRenderers();
    }
#endif
}

#ifdef USE_OPTIX6
void DICOMPlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_VOLUME, DICOM_generated_Volume_cu_ptx},
    };
    OptiXContext &context = OptiXContext::get();
    for (const auto &renderer : renderers)
    {
        PLUGIN_REGISTER_RENDERER(renderer.first);
        const std::string ptx = renderer.second;

        auto osp = std::make_shared<OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(ptx, "closest_hit_radiance");
        osp->any_hit = context.getOptixContext()->createProgramFromPTXString(ptx, "any_hit_shadow");

        context.addRenderer(renderer.first, osp);
    }
}
#endif

void DICOMPlugin::_createRenderers()
{
    auto &engine = _api->getEngine();
    _addVolumeRenderer(engine);
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