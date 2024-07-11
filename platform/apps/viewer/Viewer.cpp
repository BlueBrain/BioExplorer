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
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>

namespace core
{
Viewer::Viewer(Core& core)
    : BaseWindow(core)
{
}

void Viewer::display()
{
    const auto& pm = _core.getParametersManager();
    std::stringstream ss;
    ss << "BioExplorer Viewer";
    const auto animationFrame = pm.getAnimationParameters().getFrame();
    const auto engineName = pm.getApplicationParameters().getEngine();
    ss << " [" << engineName << "]";
    if (animationFrame != std::numeric_limits<uint32_t>::max())
        ss << " (frame " << animationFrame << ")";
    if (_core.getParametersManager().getApplicationParameters().isBenchmarking())
        ss << " @ " << _timer.perSecondSmoothed() << " FPS";
    setTitle(ss.str());

    BaseWindow::display();
}
} // namespace core
