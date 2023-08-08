/*
 * Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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

#include "OptiXOrthographicCamera.h"

#include "OptiXCamera.h"
#include "OptiXContext.h"

#include <platform/core/common/Logs.h>

#include <platform/engines/optix6/OptiX6Engine_generated_Constantbg.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_OrthographicCamera.cu.ptx.h>

const std::string PTX_ORTHOGRAPHIC_CAMERA = OptiX6Engine_generated_OrthographicCamera_cu_ptx;
const std::string PTX_MISS = OptiX6Engine_generated_Constantbg_cu_ptx;

namespace core
{
OptiXOrthographicCamera::OptiXOrthographicCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_ORTHOGRAPHIC_CAMERA, CUDA_FUNC_ORTHOGRAPHIC_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_ORTHOGRAPHIC_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXOrthographicCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    const auto position = camera.getPosition();
    const auto target = camera.getTarget();
    const auto orientation = camera.getOrientation();

    const auto height = camera.getPropertyOrValue<double>(CONTEXT_CAMERA_HEIGHT, 1.f);
    const auto aspect = camera.getPropertyOrValue<double>(CONTEXT_CAMERA_ASPECT, 1.f);

    const Vector3d dir = normalize(target - position);

    Vector3d u = normalize(cross(dir, orientation * UP_VECTOR));
    Vector3d v = cross(u, dir);

    u *= height * aspect;
    v *= height;

    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(position.x, position.y, position.z);
    context[CONTEXT_CAMERA_DIR]->setFloat(dir.x, dir.y, dir.z);
    context[CONTEXT_CAMERA_BAD_COLOR]->setFloat(1.f, 0.f, 1.f);
}
} // namespace core
