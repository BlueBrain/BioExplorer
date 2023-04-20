/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <engines/optix6/braynsOptix6Engine_generated_Constantbg.cu.ptx.h>
#include <engines/optix6/braynsOptix6Engine_generated_OrthographicCamera.cu.ptx.h>

const std::string CUDA_ORTHOGRAPHIC_CAMERA = braynsOptix6Engine_generated_OrthographicCamera_cu_ptx;
const std::string CUDA_MISS = braynsOptix6Engine_generated_Constantbg_cu_ptx;

const std::string CUDA_FUNC_ORTHOGRAPHIC_CAMERA = "orthographicCamera";
const std::string CUDA_ATTR_CAMERA_DIR = "dir";
const std::string CUDA_ATTR_CAMERA_HEIGHT = "height";

namespace brayns
{
OptiXOrthographicCamera::OptiXOrthographicCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram =
        context->createProgramFromPTXString(CUDA_ORTHOGRAPHIC_CAMERA, CUDA_FUNC_ORTHOGRAPHIC_CAMERA);
    _missProgram = context->createProgramFromPTXString(CUDA_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(CUDA_ORTHOGRAPHIC_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXOrthographicCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    const auto pos = camera.getPosition();
    const auto up = glm::rotate(camera.getOrientation(), Vector3d(0, 1, 0));

    const auto height = camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_HEIGHT, 1.0);
    const auto aspect = camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_ASPECT, 1.0);

    Vector3d dir = normalize(camera.getTarget() - pos);
    Vector3d pos_du = normalize(cross(dir, up));
    Vector3d pos_dv = cross(pos_du, dir);

    pos_du *= height * aspect;
    pos_dv *= height;

    const Vector3d pos_00 = pos - 0.5 * pos_du - 0.5 * pos_dv;

    context[CUDA_ATTR_CAMERA_W]->setFloat(pos_00.x, pos_00.y, pos_00.z);
    context[CUDA_ATTR_CAMERA_U]->setFloat(pos_du.x, pos_du.y, pos_du.z);
    context[CUDA_ATTR_CAMERA_V]->setFloat(pos_dv.x, pos_dv.y, pos_dv.z);

    context[CUDA_ATTR_CAMERA_EYE]->setFloat(pos.x, pos.y, pos.z);
    context[CUDA_ATTR_CAMERA_DIR]->setFloat(dir.x, dir.y, dir.z);
    context[CUDA_ATTR_CAMERA_HEIGHT]->setFloat(height);
    context[CUDA_ATTR_CAMERA_BAD_COLOR]->setFloat(1.f, 0.f, 1.f);
    context[CUDA_ATTR_CAMERA_OFFSET]->setFloat(0, 0);
}
} // namespace brayns
