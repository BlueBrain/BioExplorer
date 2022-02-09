/* Copyright (c) 2015-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#pragma once

#include <plugin/api/Params.h>
#include <plugin/io/handlers/FieldHandler.h>
#include <plugin/io/handlers/OctreeFieldHandler.h>

#include <brayns/pluginapi/ExtensionPlugin.h>

namespace fieldrenderer
{
/**
 * @brief This class implements the FieldRendererPlugin plugin
 */
class FieldRendererPlugin : public brayns::ExtensionPlugin
{
public:
    FieldRendererPlugin(const std::string& connectionString,
                        const std::string& schema, const bool useOctree,
                        const bool loadGeometry, const bool useCompartments);

    void init() final;

private:
    Response _version() const;

    // Fields
    void _attachFieldsHandler();
    Response _setSimulationParameters(const SimulationParameters& payload);

    std::string _uri;
    std::string _schema;
    bool _useOctree{false};
    bool _loadGeometry{false};
    bool _useCompartments{false};
};
} // namespace fieldrenderer
