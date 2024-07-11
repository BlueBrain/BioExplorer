/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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
