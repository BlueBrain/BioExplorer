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

#include <platform/core/common/Timer.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Renderer.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

TEST_CASE("default_scene_benchmark")
{
    const char* argv[] = {"core"};
    core::Core core(1, argv);

    uint64_t reference, shadowIntensity, softShadowStrength, ambientOcclusion, allOptions;

    // Set default rendering parameters
    core::ParametersManager& params = core.getParametersManager();
    params.getRenderingParameters().setSamplesPerPixel(32);
    core.commit();

    // Start timer
    core::Timer timer;
    timer.start();
    core.render();
    timer.stop();
    reference = timer.milliseconds();

    auto& renderer = core.getEngine().getRenderer();

    // Shadows
    auto props = renderer.getPropertyMap();
    props.updateProperty(RENDERER_PROPERTY_SHADOW_INTENSITY.name, 1.);
    renderer.updateProperties(props);
    core.commit();

    timer.start();
    core.render();
    timer.stop();
    shadowIntensity = timer.milliseconds();

    // Shadows
    float t = float(shadowIntensity) / float(reference);
    CHECK_MESSAGE(t < 1.65f, "Shadows cost. expected: 165%");

    props.updateProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), 1.);
    renderer.updateProperties(props);
    core.commit();

    timer.start();
    core.render();
    timer.stop();
    softShadowStrength = timer.milliseconds();

    // Soft shadowIntensity
    t = float(softShadowStrength) / float(reference);
    CHECK_MESSAGE(t < 1.85f, "Soft shadowIntensity cost. expected: 185%");

    // Ambient occlustion
    props.updateProperty(RENDERER_PROPERTY_SHADOW_INTENSITY.name, 0.);
    props.updateProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), , 0.);
    props.updateProperty("aoWeight", 1.);
    renderer.updateProperties(props);
    core.commit();

    timer.start();
    core.render();
    timer.stop();
    ambientOcclusion = timer.milliseconds();

    // Ambient occlusion
    t = float(ambientOcclusion) / float(reference);
    CHECK_MESSAGE(t < 2.5f, "Ambient occlusion cost. expected: 250%");

    // All options
    props.updateProperty(RENDERER_PROPERTY_SHADOW_INTENSITY.name, 1.);
    props.updateProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), , 1.);
    props.updateProperty("aoWeight", 1.);
    renderer.updateProperties(props);
    core.commit();

    timer.start();
    core.render();
    timer.stop();
    allOptions = timer.milliseconds();

    // All options
    t = float(allOptions) / float(reference);
    CHECK_MESSAGE(t < 3.5f, "All options cost. expected: 350%");
}
