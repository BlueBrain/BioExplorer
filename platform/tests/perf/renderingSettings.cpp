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
