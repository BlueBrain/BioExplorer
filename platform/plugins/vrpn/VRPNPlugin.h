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

#include <Defines.h>

#include <platform/core/common/Timer.h>
#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

#include <vrpn_Analog.h>
#include <vrpn_Tracker.h>

#ifdef BRAYNSVRPN_USE_LIBUV
#include <uv.h>
#endif

namespace core
{
struct VrpnStates
{
    float axisX = 0.0f;
    float axisZ = 0.0f;
    glm::quat flyStickOrientation;
};

class VRPNPlugin : public ExtensionPlugin
{
public:
    VRPNPlugin(const std::string& vrpnName);
    ~VRPNPlugin();

    void init() final;

    void preRender() final;

#ifdef BRAYNSVRPN_USE_LIBUV
    void resumeRenderingIfTrackerIsActive();
#endif

private:
    std::unique_ptr<vrpn_Tracker_Remote> _vrpnTracker;
    std::unique_ptr<vrpn_Analog_Remote> _vrpnAnalog;
    const std::string _vrpnName;
    Timer _timer;
    VrpnStates _states;

#ifdef BRAYNSVRPN_USE_LIBUV
    struct LibuvDeleter
    {
        void operator()(uv_timer_t* timer)
        {
            uv_timer_stop(timer);
            uv_close(reinterpret_cast<uv_handle_t*>(timer), [](uv_handle_t* handle) { delete handle; });
        }
    };
    std::unique_ptr<uv_timer_t, LibuvDeleter> _idleTimer;

    void _setupIdleTimer();
#endif
};
} // namespace core
