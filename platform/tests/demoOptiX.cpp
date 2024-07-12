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

    CHECK(compareTestImage("testdemoOptiX.png", core.getEngine().getFrameBuffer()));
}
