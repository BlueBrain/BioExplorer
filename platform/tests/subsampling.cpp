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

#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("subsampling_buffer_size")
{
    const char* argv[] = {"subsampling", "--window-size", "400", "200", "--subsampling", "4", "demo"};
    const int argc = sizeof(argv) / sizeof(char*);
    core::Core core(argc, argv);

    core.commitAndRender();
    CHECK_EQ(core.getEngine().getFrameBuffer().getSize(), core::Vector2ui(100, 50));

    core.commitAndRender();
    CHECK_EQ(core.getEngine().getFrameBuffer().getSize(), core::Vector2ui(400, 200));
}

TEST_CASE("no_subsampling_needed")
{
    const char* argv[] = {"subsampling", "--window-size", "400", "200", "--samples-per-pixel", "2", "demo"};
    const int argc = sizeof(argv) / sizeof(char*);
    core::Core core(argc, argv);

    core.commitAndRender();
    CHECK_EQ(core.getEngine().getFrameBuffer().getSize(), core::Vector2ui(400, 200));

    core.commitAndRender();
    CHECK_EQ(core.getEngine().getFrameBuffer().getSize(), core::Vector2ui(400, 200));
}
