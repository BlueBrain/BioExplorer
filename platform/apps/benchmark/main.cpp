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

#include <platform/core/Core.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/Timer.h>
#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

int main(int argc, const char** argv)
{
    try
    {
        const size_t nbFrames = 100;

        core::Timer timer;

        timer.start();
        core::Core core(argc, argv);
        timer.stop();

        CORE_INFO("[PERF] Scene initialization took " << timer.milliseconds() << " milliseconds");

        auto& engine = core.getEngine();
        auto& scene = engine.getScene();
        const auto bounds = scene.getBounds();
        const double radius = glm::compMax(bounds.getSize());
        timer.start();
        for (size_t frame = 0; frame < nbFrames; ++frame)
        {
            const core::Vector3d& center = bounds.getCenter();
            const auto quat = glm::angleAxis(frame * M_PI / 180.0, core::Vector3d(0.0, 1.0, 0.0));
            const core::Vector3d dir = glm::rotate(quat, core::Vector3d(0, 0, -1));
            engine.getCamera().set(center + radius * -dir, quat);
            core.commitAndRender();
        }
        timer.stop();

        CORE_INFO("[PERF] Rendering " << nbFrames << " frames took " << timer.milliseconds() << " milliseconds");
        CORE_INFO("[PERF] Frames per second: " << nbFrames / timer.seconds());
    }
    catch (const std::runtime_error& e)
    {
        CORE_ERROR(e.what());
        return 1;
    }
    return 0;
}
