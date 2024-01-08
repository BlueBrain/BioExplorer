/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <platform/core/common/Types.h>

namespace core
{
class Engine;

/**
 * Defines the abstract representation of an extension plug-in. What we mean by
 * extension is a set a functionalities that are not provided by the core of the
 * application. For example, exposing a REST interface via HTTP, or streaming
 * images to an distant display.
 *
 * For a plugin to be loaded dynamically at runtime, the following function
 * must be available in the library:
 *
 * @code
 * extern "C" core::ExtensionPlugin* brayns_plugin_create(int argc, const
 * char** argv)
 * @endcode
 *
 * It must return the instance of the plugin, and from hereon Core owns the
 * plugin and calls preRender() and postRender() accordingly.
 * In the shutdown sequence of Core, the plugin will be destructed properly.
 */
class ExtensionPlugin
{
public:
    virtual ~ExtensionPlugin() = default;

    /**
     * Called from Core::Core right after the engine has been created
     */
    virtual void init() {}
    /**
     * Called from Core::preRender() to prepare the engine based on the
     * plugins' need for an upcoming render().
     */
    virtual void preRender() {}
    /** Called from Core::postRender() after render() has finished. */
    virtual void postRender() {}

protected:
    PluginAPI* _api{nullptr};
    friend class PluginManager;
};
} // namespace core
