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
