/*
 * Copyright (c) 2018-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "Defines.h"

#include "plugin/common/OpenDeckParameters.h"

#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace core
{
class OpenDeckPlugin : public ExtensionPlugin
{
public:
    OpenDeckPlugin(const OpenDeckParameters& params);

    void init() final;

private:
    Vector2ui _wallRes;
    Vector2ui _floorRes;
    OpenDeckParameters _params;
};
} // namespace core
