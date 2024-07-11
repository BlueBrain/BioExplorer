/*
    Copyright 2015 - 2018 Blue Brain Project / EPFL

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

#ifndef ROCKETSPLUGIN_H
#define ROCKETSPLUGIN_H

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace core
{
/**
   The RocketsPlugin is in charge of exposing a both an http/REST interface to
   the outside world. The http server is configured according
   to the --http-server parameter provided by ApplicationParameters.
 */
class RocketsPlugin : public ExtensionPlugin
{
public:
    ~RocketsPlugin();
    void init() final;

    /**
     * In case no event loop is available, this processes in- and outgoing HTTP
     * and websocket messages.
     *
     * Otherwise, this is a NOP as the incoming message processing is done by
     * the SocketListener.
     */
    void preRender() final;

    /**
     * Enqueue modified and registered objects for broadcast that have changed
     * after the rendering is finished (framebuffer).
     */
    void postRender() final;

private:
    class Impl;
    std::shared_ptr<Impl> _impl;
};
} // namespace core

#endif
