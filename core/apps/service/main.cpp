/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <core/brayns/Brayns.h>
#include <core/brayns/common/Logs.h>
#include <core/brayns/common/Types.h>
#include <core/brayns/engineapi/Engine.h>
#include <core/brayns/engineapi/FrameBuffer.h>
#include <core/brayns/parameters/ParametersManager.h>

#include <chrono>
#include <thread>

int main(int argc, const char** argv)
{
    try
    {
        brayns::Brayns brayns(argc, argv);

        const auto& engine = brayns.getEngine();
        const auto& fbs = engine.getFrameBuffers();
        const auto& rp = engine.getParametersManager().getRenderingParameters();
        const auto fb = fbs[0];
        if (!fb)
            CORE_THROW("Engine has no frame buffer");
        while (true)
        {
            if (fb->numAccumFrames() < rp.getMaxAccumFrames())
                brayns.commitAndRender();
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                brayns.commit();
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        CORE_ERROR(e.what());
        return 1;
    }
    return 0;
}
