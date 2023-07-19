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

#include "BlackHolePlugin.h"

#include <Version.h>

#include <plugin/common/Logs.h>

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
#endif

namespace spaceexplorer
{
namespace blackhole
{
using namespace core;

const std::string PLUGIN_API_PREFIX = "bh-";

const std::string RENDERER_BLACK_HOLE = "blackhole";
const double MAX_BLACK_HOLE_SIZE = 100.0;

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
    properties.setProperty({"mainExposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"nbDisks", 20, 2, 128, {"Number of disks"}});
    properties.setProperty({"grid", false, {"Display grid"}});
    properties.setProperty({"timestamp", 0., 0., 8192., {"Timestamp"}});
    properties.setProperty({"diskRotationSpeed", 3., 1., 10., {"Disk rotation speed"}});
    properties.setProperty({"diskTextureLayers", 12, 2, 100, {"Disk texture layers"}});
    properties.setProperty({"blackHoleSize", 0.3, 0.1, MAX_BLACK_HOLE_SIZE, {"Black hole size"}});
    engine.addRendererType("blackhole", properties);
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
    _api->getParametersManager().getRenderingParameters().setCurrentRenderer(RENDERER_BLACK_HOLE);
}

#ifdef USE_OPTIX6
void BlackHolePlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_BLACK_HOLE, BlackHole_generated_BlackHole_cu_ptx},
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