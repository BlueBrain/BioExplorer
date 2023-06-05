/* Copyright (c) 2016, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel.Nachbaur@epfl.ch
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
#include <platform/core/manipulators/InspectCenterManipulator.h>
#include <platform/core/parameters/ParametersManager.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("simple_construction")
{
    const char* argv[] = {"core"};
    CHECK_NOTHROW(core::Core(1, argv));
}

TEST_CASE("defaults")
{
    const char* argv[] = {"core", "demo"};
    const int argc = sizeof(argv) / sizeof(char*);
    core::Core core(argc, argv);

    auto& camera = core.getEngine().getCamera();
    CHECK_EQ(camera.getCurrentType(), "perspective");
    CHECK_EQ(camera.getPosition(), core::Vector3d(0.5, 0.5, 1.5));
    CHECK_EQ(camera.getOrientation(), core::Quaterniond(1, 0, 0, 0));

    auto& manipulator = core.getCameraManipulator();
    CHECK(dynamic_cast<core::InspectCenterManipulator*>(&manipulator));

    auto& fb = core.getEngine().getFrameBuffer();
    CHECK(!fb.getColorBuffer());
    CHECK_EQ(fb.getColorDepth(), 4);
    CHECK(!fb.getDepthBuffer());
    CHECK_EQ(fb.getSize(), core::Vector2ui(800, 600));

    auto& pm = core.getParametersManager();
    const auto& appParams = pm.getApplicationParameters();
    CHECK(appParams.getEngine() == "ospray");
    CHECK(appParams.getOsprayModules().empty());
    CHECK_EQ(appParams.getWindowSize(), core::Vector2ui(800, 600));
    CHECK(!appParams.isBenchmarking());
    CHECK_EQ(appParams.getJpegCompression(), 90);
    CHECK_EQ(appParams.getImageStreamFPS(), 60);

    const auto& renderParams = pm.getRenderingParameters();
    CHECK_EQ(renderParams.getCurrentCamera(), "perspective");
    CHECK_EQ(renderParams.getCurrentRenderer(), "basic");
    CHECK_EQ(renderParams.getCameras().size(), 5);
    CHECK_EQ(renderParams.getRenderers().size(), 5);
    CHECK_EQ(renderParams.getSamplesPerPixel(), 1);
    CHECK_EQ(renderParams.getBackgroundColor(), core::Vector3d(0, 0, 0));

    const auto& geomParams = pm.getGeometryParameters();
    CHECK(geomParams.getColorScheme() == core::ProteinColorScheme::none);
    CHECK(geomParams.getGeometryQuality() == core::GeometryQuality::high);

    const auto& animParams = pm.getAnimationParameters();
    CHECK_EQ(animParams.getFrame(), 0);

    const auto& volumeParams = pm.getVolumeParameters();
    CHECK_EQ(volumeParams.getDimensions(), core::Vector3ui(0, 0, 0));
    CHECK_EQ(volumeParams.getElementSpacing(), core::Vector3d(1., 1., 1.));
    CHECK_EQ(volumeParams.getOffset(), core::Vector3d(0., 0., 0.));

    auto& scene = core.getEngine().getScene();
    core::Boxd defaultBoundingBox;
    defaultBoundingBox.merge(core::Vector3d(0, 0, 0));
    defaultBoundingBox.merge(core::Vector3d(1, 1, 1));
    CHECK_EQ(scene.getBounds(), defaultBoundingBox);
    CHECK(geomParams.getMemoryMode() == core::MemoryMode::shared);
}

TEST_CASE("bvh_type")
{
    const char* argv[] = {
        "core",  "demo", "--default-bvh-flag", "robust", "--default-bvh-flag",
        "compact",
    };
    const int argc = sizeof(argv) / sizeof(char*);
    core::Core core(argc, argv);

    auto model = core.getEngine().getScene().getModel(0);
    const auto& bvhFlags = model->getModel().getBVHFlags();

    CHECK(bvhFlags.count(core::BVHFlag::robust) > 0);
    CHECK(bvhFlags.count(core::BVHFlag::compact) > 0);
}
