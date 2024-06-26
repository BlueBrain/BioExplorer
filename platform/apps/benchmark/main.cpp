/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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
