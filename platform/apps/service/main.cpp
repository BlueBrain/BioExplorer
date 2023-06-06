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
#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/parameters/ParametersManager.h>

#include <chrono>
#include <thread>

int main(int argc, const char** argv)
{
    try
    {
        core::Core core(argc, argv);

        const auto& engine = core.getEngine();
        const auto& fbs = engine.getFrameBuffers();
        const auto& rp = engine.getParametersManager().getRenderingParameters();
        const auto fb = fbs[0];
        if (!fb)
            CORE_THROW("Engine has no frame buffer");
        while (true)
        {
            if (fb->numAccumFrames() < rp.getMaxAccumFrames())
                core.commitAndRender();
            else
                core.commit();
        }
    }
    catch (const std::runtime_error& e)
    {
        CORE_ERROR(e.what());
        return 1;
    }
    return 0;
}
