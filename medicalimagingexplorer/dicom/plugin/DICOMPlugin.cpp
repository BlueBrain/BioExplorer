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
#include <platform/core/common/Progress.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

namespace medicalimagingexplorer
{
namespace dicom
{
#define REGISTER_LOADER(LOADER, FUNC) registry.registerLoader({std::bind(&LOADER::getSupportedDataTypes), FUNC});

using namespace core;

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