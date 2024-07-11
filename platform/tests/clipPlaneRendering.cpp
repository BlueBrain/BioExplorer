/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

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

#include "PDiffHelpers.h"

#include <platform/core/Core.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Scene.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

class Demo
{
public:
    Demo()
        : _argv{CAMERA_PROPERTY_CLIPPING_PLANES, "demo", "--disable-accumulation", "--window-size", "50", "50"}
        , _core(_argv.size(), _argv.data())
    {
        instance = &_core;
    }

    static core::Core* instance;

private:
    std::vector<const char*> _argv;
    core::Core _core;
};
core::Core* Demo::instance = nullptr;

void testClipping(core::Core& core, bool orthographic = false)
{
    const std::string original = orthographic ? "demo_ortho.png" : "snapshot.png";

    const std::string clipped = orthographic ? "demo_clipped_ortho.png" : "demo_clipped_perspective.png";

    auto& engine = core.getEngine();
    auto& scene = engine.getScene();
    auto& camera = engine.getCamera();

    auto position = scene.getBounds().getCenter();
    position.z += glm::compMax(scene.getBounds().getSize());

    camera.setInitialState(position, glm::identity<core::Quaterniond>());

    if (orthographic)
        camera.setCurrentType("orthographic");
    else
        camera.setCurrentType("perspective");
    core.commitAndRender();
    CHECK(compareTestImage(original, engine.getFrameBuffer()));

    auto id1 = scene.addClipPlane({{1.0, 0.0, 0.0, -0.5}});
    auto id2 = scene.addClipPlane({{0.0, -1.0, 0.0, 0.5}});
    core.commitAndRender();
    CHECK(compareTestImage(clipped, engine.getFrameBuffer()));

    scene.removeClipPlane(id1);
    scene.removeClipPlane(id2);
    core.commitAndRender();
    CHECK(compareTestImage(original, engine.getFrameBuffer()));

    id1 = scene.addClipPlane({{1.0, 0.0, 0.0, -0.5}});
    id2 = scene.addClipPlane({{0.0, 1.0, 0.0, 0.5}});
    scene.getClipPlane(id2)->setPlane({{0.0, -1.0, 0.0, 0.5}});
    core.commitAndRender();
    CHECK(compareTestImage(clipped, engine.getFrameBuffer()));

    scene.removeClipPlane(id1);
    scene.removeClipPlane(id2);
}

TEST_CASE_FIXTURE(Demo, "perspective")
{
    testClipping(*Demo::instance);
}

TEST_CASE_FIXTURE(Demo, "orthographic")
{
    testClipping(*Demo::instance, true);
}
