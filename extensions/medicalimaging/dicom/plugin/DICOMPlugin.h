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

#pragma once

#include <plugin/io/DICOMLoader.h>

#include <Defines.h>

#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace medicalimagingexplorer
{
namespace dicom
{
/**
 * @brief The DICOM plugin class manages the loading of DICOM datasets
 */
class DICOMPlugin : public core::ExtensionPlugin
{
public:
    DICOMPlugin(core::PropertyMap&& dicomParams);

    void init() final;

private:
    void _createRenderers();
#ifdef USE_OPTIX6
    void _createOptiXRenderers();
#endif

    core::PropertyMap _dicomParams;
    bool _dirty{false};
};
} // namespace dicom
} // namespace medicalimagingexplorer