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

#pragma once

#include <plugin/io/DICOMLoader.h>

#include <Defines.h>

#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace medicalimagingexplorer
{
namespace dicom
{
using namespace core;

/**
 * @brief The DICOM plugin class manages the loading of DICOM datasets
 */
class DICOMPlugin : public ExtensionPlugin
{
public:
    DICOMPlugin(PropertyMap&& dicomParams);

    void init() final;

private:
    void _createRenderers();
#ifdef USE_OPTIX6
    void _createOptiXRenderers();
#endif

    PropertyMap _dicomParams;
    bool _dirty{false};
};
} // namespace dicom
} // namespace medicalimagingexplorer