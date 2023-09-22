/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "OptiXCamera.h"
#include "OptiXCameraProgram.h"
#include "OptiXUtils.h"

#include <platform/core/common/Logs.h>

namespace core
{
namespace engine
{
namespace optix
{
void OptiXCamera::commit()
{
    if (_currentCamera != getCurrentType())
    {
        _currentCamera = getCurrentType();
        OptiXContext::get().setCamera(_currentCamera);
    }

    auto cameraProgram = OptiXContext::get().getCamera(_currentCamera);
    auto context = OptiXContext::get().getOptixContext();
    toOptiXProperties(getPropertyMap());
    cameraProgram->commit(*this, context);
}
} // namespace optix
} // namespace engine
} // namespace core