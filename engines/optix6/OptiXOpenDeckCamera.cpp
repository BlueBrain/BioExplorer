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

#include "OptiXOpenDeckCamera.h"

#include "OptiXCamera.h"
#include "OptiXContext.h"

#include <engines/optix6/braynsOptix6Engine_generated_Constantbg.cu.ptx.h>
#include <engines/optix6/braynsOptix6Engine_generated_OpenDeckCamera.cu.ptx.h>

const std::string PTX_OPENDECK_CAMERA = braynsOptix6Engine_generated_OpenDeckCamera_cu_ptx;
const std::string PTX_MISS = braynsOptix6Engine_generated_Constantbg_cu_ptx;

namespace brayns
{
OptiXOpenDeckCamera::OptiXOpenDeckCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_OPENDECK_CAMERA, CUDA_FUNC_OPENDECK_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_OPENDECK_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXOpenDeckCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    const auto pos = camera.getPosition();

    const Vector3d u = normalize(glm::rotate(camera.getOrientation(), Vector3d(1, 0, 0)));
    const Vector3d v = normalize(glm::rotate(camera.getOrientation(), Vector3d(0, 1, 0)));
    const Vector3d w = normalize(glm::rotate(camera.getOrientation(), Vector3d(0, 0, 1)));

    const auto headPos = camera.getPropertyOrValue<std::array<double, 3>>("headPosition", {{0., 0., 0.}});
    const auto headRotation = camera.getPropertyOrValue<std::array<double, 4>>("headRotation", {{0., 0., 0., 1.}});
    const auto headUVec = glm::rotate(Quaterniond(headRotation[3], headRotation[0], headRotation[1], headRotation[2]),
                                      Vector3d(1., 0., 0.));

    context[CONTEXT_CAMERA_SEGMENT_ID]->setUint(camera.getPropertyOrValue<int>("segmentId", 0));
    context[CONTEXT_CAMERA_HALF_IPD]->setFloat(camera.getPropertyOrValue<double>("interpupillaryDistance", 0.065) /
                                               2.0);
    context[CONTEXT_CAMERA_HEAD_POS]->setFloat(headPos[0], headPos[1], headPos[2]);
    context[CONTEXT_CAMERA_HEAD_UVEC]->setFloat(headUVec.x, headUVec.y, headUVec.z);
    context[CONTEXT_CAMERA_EYE]->setFloat(pos.x, pos.y, pos.z);
    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(w.x, w.y, w.z);
    context[CONTEXT_CAMERA_APERTURE_RADIUS]->setFloat(camera.getPropertyOrValue<double>("apertureRadius", 0.0));
    context[CONTEXT_CAMERA_FOCAL_SCALE]->setFloat(camera.getPropertyOrValue<double>("focusDistance", 1.0));
    context[CONTEXT_CAMERA_BAD_COLOR]->setFloat(1.f, 0.f, 1.f);
    // context[CONTEXT_CAMERA_OFFSET]->setFloat(0, 0);
}
} // namespace brayns
