/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

#include "DeflectParameters.h"

#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace core
{
class DeflectPlugin : public ExtensionPlugin
{
public:
    DeflectPlugin(DeflectParameters&& params);
    void init() final;

    /** Handle stream setup and incoming events. */
    void preRender() final;

    /** Send rendered frame. */
    void postRender() final;

private:
    class Impl;
    std::shared_ptr<Impl> _impl;

    DeflectParameters _params;
};
} // namespace core
