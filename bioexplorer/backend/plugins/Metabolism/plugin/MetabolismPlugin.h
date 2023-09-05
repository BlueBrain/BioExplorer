/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
