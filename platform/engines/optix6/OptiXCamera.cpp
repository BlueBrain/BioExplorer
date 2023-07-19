/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
void OptiXCamera::commit()
{
    if (_currentCamera != getCurrentType())
    {
        _currentCamera = getCurrentType();
        OptiXContext::get().setCamera(_currentCamera);
    }

    auto cameraProgram = OptiXContext::get().getCamera(_currentCamera);

    auto context = OptiXContext::get().getOptixContext();

    cameraProgram->commit(*this, context);

    RT_DESTROY(_clipPlanesBuffer);

    const size_t numClipPlanes = _clipPlanes.size();
    if (numClipPlanes > 0)
    {
        Vector4fs buffer;
        buffer.reserve(numClipPlanes);
        for (const auto& clipPlane : _clipPlanes)
            buffer.push_back({static_cast<float>(clipPlane[0]), static_cast<float>(clipPlane[1]),
                              static_cast<float>(clipPlane[2]), static_cast<float>(clipPlane[3])});

        _clipPlanesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, numClipPlanes);
        memcpy(_clipPlanesBuffer->map(), buffer.data(), numClipPlanes * sizeof(Vector4f));
        _clipPlanesBuffer->unmap();
    }
    else
    {
        // Create empty buffer to avoid unset variable exception in cuda
        _clipPlanesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1);
    }

    context[CONTEXT_CLIP_PLANES]->setBuffer(_clipPlanesBuffer);
    context[CONTEXT_NB_CLIP_PLANES]->setUint(numClipPlanes);
}

} // namespace core
