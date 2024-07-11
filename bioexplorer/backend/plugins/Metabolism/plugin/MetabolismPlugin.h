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

#include <plugin/api/Params.h>
#include <plugin/io/handlers/MetabolismHandler.h>

#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
namespace metabolism
{
/**
 * @brief This class implements the MetabolismPlugin plugin
 */
class MetabolismPlugin : public core::ExtensionPlugin
{
public:
    MetabolismPlugin(int argc, char **argv);

    void init() final;

private:
    void _parseCommandLineArguments(int argc, char **argv);

    // Metabolism
    bioexplorer::details::Response _attachHandler(const AttachHandlerDetails &payload);

    // Command line arguments
    CommandLineArguments _commandLineArguments;
};
} // namespace metabolism
} // namespace bioexplorer
