/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "Viewer.h"
#include <platform/core/Core.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/Types.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>

int main(int argc, const char** argv)
{
    try
    {
        core::Core core(argc, argv);
        core::initGLUT(&argc, argv);
        core::Viewer viewer(core);
        CORE_INFO("Initializing Application...");
        const core::Vector2ui& size = core.getParametersManager().getApplicationParameters().getWindowSize();

        viewer.create("Core Viewer", size.x, size.y);
        core::runGLUT();
    }
    catch (const std::runtime_error& e)
    {
        CORE_ERROR(e.what());
        return 1;
    }
    return 0;
}
