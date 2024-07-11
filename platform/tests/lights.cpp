/*
    Copyright 2019 - 0211 Blue Brain Project / EPFL

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

#include <platform/core/common/light/Light.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/LightManager.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "PDiffHelpers.h"

namespace
{
const auto YELLOW = core::Vector3f(1.0f, 1.0f, 0.0f);
const auto BLUE = core::Vector3f(0.0f, 0.0f, 1.0f);

const float lampHeight = 0.99f;
const float lampWidth = 0.15f;

const core::Vector3f lampCentre = {0.5f, lampHeight, 0.5f};

const core::Vector3f lampPositions[4] = {{lampCentre.x - lampWidth, lampHeight, lampCentre.z - lampWidth},
                                         {lampCentre.x + lampWidth, lampHeight, lampCentre.z - lampWidth},
                                         {lampCentre.x + lampWidth, lampHeight, lampCentre.z + lampWidth},
                                         {lampCentre.x - lampWidth, lampHeight, lampCentre.z + lampWidth}};
} // namespace

TEST_CASE("render_scivis_quadlight")
{
    const char* argv[] = {RENDERER_PROPERTY_LIGHTS, "demo", "--engine", ENGINE_OSPRAY, "--renderer", "scivis",
                          "--no-head-light"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);

    core.getEngine().getScene().getLightManager().addLight(
        std::make_shared<core::QuadLight>(lampPositions[0], (lampPositions[1] - lampPositions[0]),
                                          (lampPositions[3] - lampPositions[0]), YELLOW, 1.0f, true));

    core.commitAndRender();

    CHECK(compareTestImage("testLightScivisQuadLight.png", core.getEngine().getFrameBuffer()));
}

TEST_CASE("render_scivis_spotlight")
{
    const char* argv[] = {RENDERER_PROPERTY_LIGHTS, "demo", "--engine", ENGINE_OSPRAY, "--renderer", "scivis",
                          "--no-head-light"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);

    core.getEngine().getScene().getLightManager().addLight(
        std::make_shared<core::SpotLight>(lampCentre, core::Vector3f(0, -1, 0), 90.f, 10.f, lampWidth, BLUE, 1.0f,
                                          true));
    core.commitAndRender();

    CHECK(compareTestImage("testLightScivisSpotLight.png", core.getEngine().getFrameBuffer()));
}
