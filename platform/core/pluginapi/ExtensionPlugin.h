/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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
