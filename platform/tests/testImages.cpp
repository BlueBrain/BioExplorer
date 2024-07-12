
/*
    Copyright 2016 - 2024 Blue Brain Project / EPFL

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

#include <Defines.h>

#include <platform/core/Core.h>
#include <tests/paths.h>

#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <platform/core/defines.h>
#ifdef USE_NETWORKING
#include "ClientServer.h"
#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif

#include "PDiffHelpers.h"
TEST_CASE("render_two_frames_and_compare_they_are_same")
{
    const char* argv[] = {"testImages", "--disable-accumulation", "demo"};
    const int argc = sizeof(argv) / sizeof(char*);
    core::Core core(argc, argv);

    core.commitAndRender();
    const auto oldImage = createPDiffRGBAImage(core.getEngine().getFrameBuffer());

    core.getEngine().getFrameBuffer().clear();
    core.commitAndRender();
    const auto newImage = createPDiffRGBAImage(core.getEngine().getFrameBuffer());

    CHECK(pdiff::yee_compare(*oldImage, *newImage));
}

TEST_CASE("render_xyz_and_compare")
{
    const auto path = BRAYNS_TESTDATA_MODEL_MONKEY_PATH;
    const char* argv[] = {"testImages", path, "--disable-accumulation"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);
    core.commitAndRender();
    CHECK(compareTestImage("testdataMonkey.png", core.getEngine().getFrameBuffer()));

    auto model = core.getEngine().getScene().getModel(0);
    auto props = model->getProperties();
    props.updateProperty("radius", props.getProperty<double>("radius") / 2.);
    model->setProperties(props);

    core.getEngine().getScene().markModified();

    core.commitAndRender();
    CHECK(compareTestImage("testdataMonkey_smaller.png", core.getEngine().getFrameBuffer()));
}

#ifdef USE_NETWORKING
TEST_CASE("render_xyz_change_radius_from_rockets")
{
    const auto path = BRAYNS_TESTDATA_MODEL_MONKEY_PATH;
    const std::vector<const char*> argv = {path, "--disable-accumulation"};

    ClientServer clientServer(argv);

    auto model = clientServer.getBrayns().getEngine().getScene().getModel(0);
    core::PropertyMap props;
    props.setProperty({"radius", model->getProperties().getProperty<double>("radius") / 2.});

    CHECK(
        (clientServer.makeRequest<core::ModelProperties, bool>("set-model-properties", {model->getModelID(), props})));

    clientServer.getBrayns().getEngine().getScene().markModified();

    clientServer.getBrayns().commitAndRender();
    CHECK(compareTestImage("testdataMonkey_smaller.png", clientServer.getBrayns().getEngine().getFrameBuffer()));
}
#endif

TEST_CASE("render_protein_and_compare")
{
    const char* argv[] = {"testImages", BRAYNS_TESTDATA_MODEL_PDB_PATH, "--disable-accumulation"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);
    core.commitAndRender();
    CHECK(compareTestImage("testdataProtein.png", core.getEngine().getFrameBuffer()));
}

TEST_CASE("render_protein_in_stereo_and_compare")
{
    const char* argv[] = {"testImages", BRAYNS_TESTDATA_MODEL_PDB_PATH, "--disable-accumulation", "--stereo"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);
    core.commitAndRender();
    CHECK(compareTestImage("testdataProtein_left_eye.png", *core.getEngine().getFrameBuffers()[0]));
    CHECK(compareTestImage("testdataProtein_right_eye.png", *core.getEngine().getFrameBuffers()[1]));
}

#ifdef USE_ASSIMP
TEST_CASE("render_ply_and_compare")
{
    const auto path = BRAYNS_TESTDATA_MODEL_LUCY_PATH;
    const char* argv[] = {"testImages", path, "--disable-accumulation"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);
    core.commitAndRender();
    CHECK(compareTestImage("testdataLucy.png", core.getEngine().getFrameBuffer()));
}
#endif

#if PLATFORM_USE_LIBARCHIVE
TEST_CASE("render_capsule_and_compare")
{
    const char* argv[] = {"testImages", BRAYNS_TESTDATA_MODEL_CAPSULE_PATH, "--samples-per-pixel", "128"};
    const int argc = sizeof(argv) / sizeof(char*);

    core::Core core(argc, argv);
    core.commitAndRender();
    CHECK(compareTestImage("testCapsule.png", core.getEngine().getFrameBuffer()));
}
#endif
