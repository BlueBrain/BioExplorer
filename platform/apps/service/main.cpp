/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <platform/core/Core.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/Timer.h>
#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/parameters/ParametersManager.h>

#include <uvw.hpp>

#include <chrono>
#include <thread>

using namespace core;

class Service
{
public:
    Service(int argc, const char** argv)
        : _renderingDone{_mainLoop->resource<uvw::AsyncHandle>()}
        , _eventRendering{_mainLoop->resource<uvw::IdleHandle>()}
        , _accumRendering{_mainLoop->resource<uvw::IdleHandle>()}
        , _checkIdleRendering{_mainLoop->resource<uvw::CheckHandle>()}
        , _sigintHandle{_mainLoop->resource<uvw::SignalHandle>()}
        , _triggerRendering{_renderLoop->resource<uvw::AsyncHandle>()}
        , _stopRenderThread{_renderLoop->resource<uvw::AsyncHandle>()}
    {
        _checkIdleRendering->start();

        _setupMainThread();
        _setupRenderThread();

        _core = std::make_unique<Core>(argc, argv);

        // events from rockets, trigger rendering
        _core->getEngine().triggerRender = [&eventRendering = _eventRendering] { eventRendering->start(); };

        // launch first frame; after that, only events will trigger that
        _eventRendering->start();

        // stop the application on Ctrl+C
        _sigintHandle->once<uvw::SignalEvent>([&](const auto&, auto&) { this->_stopMainLoop(); });
        _sigintHandle->start(SIGINT);
    }

    void run()
    {
        // Start render & main loop
        std::thread renderThread([&_renderLoop = _renderLoop] { _renderLoop->run(); });
#if 1
        while (_core->getEngine().getKeepRunning())
        {
            _core->commit();
            _mainLoop->run<uvw::Loop::Mode::NOWAIT>();
        }
#else
        _mainLoop->run();
#endif

        // Finished
        renderThread.join();
    }

private:
    void _setupMainThread()
    {
        // triggered after rendering, send events to rockets from the main thread
        _renderingDone->on<uvw::AsyncEvent>([&Core = _core](const auto&, auto&) { Core->postRender(); });

        // render or data load trigger from events
        _eventRendering->on<uvw::IdleEvent>(
            [&](const auto&, auto&)
            {
                _eventRendering->stop();
                _accumRendering->stop();
                _timeSinceLastEvent.start();

                // stop event loop(s) and exit application
                if (!_core->getEngine().getKeepRunning())
                {
                    this->_stopMainLoop();
                    return;
                }

                // rendering
                if (_core->commit())
                    _triggerRendering->send();
            });

        // start accum rendering when we have no more other events
        _checkIdleRendering->on<uvw::CheckEvent>([&accumRendering = _accumRendering](const auto&, auto&)
                                                 { accumRendering->start(); });

        // accumulation rendering on idle; re-triggered by _checkIdleRendering
        _accumRendering->on<uvw::IdleEvent>(
            [&](const auto&, auto&)
            {
                if (_timeSinceLastEvent.elapsed() < _idleRenderingDelay)
                    return;

                if (_core->getEngine().continueRendering() && _core->commit())
                    _triggerRendering->send();

                _accumRendering->stop();
            });
    }

    void _setupRenderThread()
    {
        // rendering, triggered from main thread
        _triggerRendering->on<uvw::AsyncEvent>(
            [&](const auto&, auto&)
            {
                _core->render();
                _renderingDone->send();

                if (_core->getParametersManager().getApplicationParameters().isBenchmarking())
                    std::cout << _core->getEngine().getStatistics().getFPS() << " fps" << std::endl;
            });

        // stop render loop, triggered from main thread
        _stopRenderThread->once<uvw::AsyncEvent>([&](const auto&, auto&) { this->_stopRenderLoop(); });
    }

    void _stopMainLoop()
    {
        // send stop render loop message
        _core->getEngine().triggerRender = [] {};
        _stopRenderThread->send();

        // close all main loop resources to avoid memleaks
        _renderingDone->close();
        _eventRendering->close();
        _accumRendering->close();
        _checkIdleRendering->close();
        _sigintHandle->close();

        _mainLoop->stop();
    }

    void _stopRenderLoop()
    {
        _triggerRendering->close();
        _stopRenderThread->close();
        _renderLoop->stop();
    }

    std::unique_ptr<core::Core> _core;

    std::shared_ptr<uvw::Loop> _mainLoop{uvw::Loop::getDefault()};
    std::shared_ptr<uvw::AsyncHandle> _renderingDone;
    std::shared_ptr<uvw::IdleHandle> _eventRendering;
    std::shared_ptr<uvw::IdleHandle> _accumRendering;
    std::shared_ptr<uvw::CheckHandle> _checkIdleRendering;
    std::shared_ptr<uvw::SignalHandle> _sigintHandle;

    std::shared_ptr<uvw::Loop> _renderLoop{uvw::Loop::create()};
    std::shared_ptr<uvw::AsyncHandle> _triggerRendering;
    std::shared_ptr<uvw::AsyncHandle> _stopRenderThread;

    const float _idleRenderingDelay{0.1f};
    Timer _timeSinceLastEvent;
};

int main(int argc, const char** argv)
{
    try
    {
        Timer timer;
        timer.start();

        Service service(argc, argv);

        service.run();

        timer.stop();
        CORE_INFO("Service was running for " << timer.seconds() << " seconds");
    }
    catch (const std::runtime_error& e)
    {
        CORE_ERROR(e.what());
        return 1;
    }
    return 0;
}
