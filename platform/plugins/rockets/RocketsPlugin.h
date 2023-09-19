/*
 * Copyright (c) 2015-2018 EPFL/Blue Brain Project
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
