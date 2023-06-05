/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: jonas.karlsson@epfl.ch
 *
 * This file is part of Core <https://github.com/BlueBrain/Core>
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

#include <platform/core/Core.h>

#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include "PDiffHelpers.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("render_demo")
{
    const char* argv[] = {"demoOptix", "demo", "--engine", "optix"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);

    const core::Vector3d rotCenter = {0.5, 0.5, 0.5};

    auto& camera = core.getEngine().getCamera();
    const auto camPos = camera.getPosition();

    camera.setOrientation(core::Quaterniond(1, 0, 0, 0));
    camera.setPosition(camPos - (rotCenter - camPos));

    core.commitAndRender();

    CHECK(compareTestImage("testdemoOptiX.png",
                           core.getEngine().getFrameBuffer()));
}
