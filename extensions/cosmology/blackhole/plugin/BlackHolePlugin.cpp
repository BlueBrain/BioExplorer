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

#include "BlackHolePlugin.h"

#include <Version.h>

#include <plugin/common/Logs.h>
#include <plugin/common/Properties.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/geometry/TriangleMesh.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <BlackHole_generated_BlackHole.cu.ptx.h>
#include <platform/engines/optix6/OptiXContext.h>
#include <platform/engines/optix6/OptiXProperties.h>
#endif

using namespace core;

namespace spaceexplorer
{
namespace blackhole
{
static const std::string PLUGIN_API_PREFIX = "bh-";
static const std::string RENDERER_BLACK_HOLE = "blackhole";

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR << e.what());      \
        ;                               \
    }

void _addBlackHoleRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'blackhole' renderer");
    PropertyMap properties;
    properties.setProperty(COMMON_PROPERTY_EXPOSURE);
    properties.setProperty(BLACK_HOLE_RENDERER_PROPERTY_NB_DISKS);
    properties.setProperty(BLACK_HOLE_RENDERER_PROPERTY_DISPLAY_GRID);
    properties.setProperty(RENDERER_PROPERTY_TIMESTAMP);
    properties.setProperty(BLACK_HOLE_RENDERER_PROPERTY_DISK_ROTATION_SPEED);
    properties.setProperty(BLACK_HOLE_RENDERER_PROPERTY_DISK_TEXTURE_LAYERS);
    properties.setProperty(BLACK_HOLE_RENDERER_PROPERTY_SIZE);
    engine.addRendererType(RENDERER_BLACK_HOLE, properties);
}

BlackHolePlugin::BlackHolePlugin()
    : ExtensionPlugin()
{
}

void BlackHolePlugin::init()
{
    auto actionInterface = _api->getActionInterface();

    if (actionInterface)
    {
        std::string entryPoint = PLUGIN_API_PREFIX + "version";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<Response>(entryPoint, [&]() { return _version(); });
    }

    auto &engine = _api->getEngine();
    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
        _createOptiXRenderers();
#endif
    _createRenderers();
    _createBoundingBox();
}

void BlackHolePlugin::_createRenderers()
{
    auto &engine = _api->getEngine();
    _addBlackHoleRenderer(engine);
    engine.setRendererType(RENDERER_BLACK_HOLE);
}

#ifdef USE_OPTIX6
void BlackHolePlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_BLACK_HOLE, BlackHole_generated_BlackHole_cu_ptx},
    };
    core::engine::optix::OptiXContext &context = core::engine::optix::OptiXContext::get();
    for (const auto &renderer : renderers)
    {
        PLUGIN_REGISTER_RENDERER(renderer.first);
        const std::string ptx = renderer.second;

        auto osp = std::make_shared<core::engine::optix::OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, core::engine::optix::OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE);
        osp->any_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, core::engine::optix::OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW);

        context.addRenderer(renderer.first, osp);
    }
}
#endif

void BlackHolePlugin::_createBoundingBox()
{
    const double blackHoleHalfSize = MAX_BLACK_HOLE_SIZE / 2.0;
    TriangleMesh mesh = createBox(Vector3d(-blackHoleHalfSize, -blackHoleHalfSize, -blackHoleHalfSize),
                                  Vector3d(blackHoleHalfSize, blackHoleHalfSize, blackHoleHalfSize));

    auto &scene = _api->getScene();
    auto model = scene.createModel();
    const size_t materialId = 0;
    const std::string name = "BlackHoleBounds";

    auto material = model->createMaterial(materialId, name);
    material->setOpacity(0.f);

    model->getTriangleMeshes()[materialId] = mesh;

    size_t numModels = scene.getNumModels();
    scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), name));
}

Response BlackHolePlugin::_version() const
{
    Response response;
    response.contents = PACKAGE_VERSION_STRING;
    return response;
}

extern "C" ExtensionPlugin *core_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("_|_|_|    _|                      _|            _|    _|            _|            ");
    PLUGIN_INFO("_|    _|  _|    _|_|_|    _|_|_|  _|  _|        _|    _|    _|_|    _|    _|_|    ");
    PLUGIN_INFO("_|_|_|    _|  _|    _|  _|        _|_|          _|_|_|_|  _|    _|  _|  _|_|_|_|  ");
    PLUGIN_INFO("_|    _|  _|  _|    _|  _|        _|  _|        _|    _|  _|    _|  _|  _|        ");
    PLUGIN_INFO("_|_|_|    _|    _|_|_|    _|_|_|  _|    _|      _|    _|    _|_|    _|    _|_|_|  ");
    PLUGIN_INFO("");

    return new BlackHolePlugin();
}

} // namespace blackhole
} // namespace spaceexplorer