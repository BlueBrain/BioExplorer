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

#include "OptiXCylindricStereoCamera.h"

#include <platform/engines/optix6/OptiXCamera.h>
#include <platform/engines/optix6/OptiXContext.h>

#include <CorePluginOpenDeck_generated_CylindricStereoCamera.cu.ptx.h>

#include <platform/engines/optix6/OptiX6Engine_generated_Constantbg.cu.ptx.h>

namespace core
{
const std::string PTX_CYLINDRIC_STEREO_CAMERA = CorePluginOpenDeck_generated_CylindricStereoCamera_cu_ptx;
const std::string PTX_MISS = OptiX6Engine_generated_Constantbg_cu_ptx;
const std::string CUDA_FUNC_OPENDECK_CAMERA = "openDeckCamera";
const std::string CONTEXT_CAMERA_SEGMENT_ID = "segmentID";
const std::string CONTEXT_CAMERA_HEAD_POSITION = "head_position";
const std::string CONTEXT_CAMERA_HEAD_ROTATION = "head_rotation";
const std::string CONTEXT_CAMERA_HEAD_UVEC = "headUVec";
const std::string CONTEXT_CAMERA_HALF_IPD = "half_ipd";
const std::string CONTEXT_CAMERA_FOCUS_DISTANCE = "focus_distance";

OptiXCylindricStereoCamera::OptiXCylindricStereoCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_CYLINDRIC_STEREO_CAMERA, CUDA_FUNC_OPENDECK_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_CYLINDRIC_STEREO_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXCylindricStereoCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    const auto pos = camera.getPosition();

    const Vector3d u = normalize(glm::rotate(camera.getOrientation(), Vector3d(1, 0, 0)));
    const Vector3d v = normalize(glm::rotate(camera.getOrientation(), Vector3d(0, 1, 0)));
    const Vector3d w = normalize(glm::rotate(camera.getOrientation(), Vector3d(0, 0, 1)));

    context[CONTEXT_CAMERA_EYE]->setFloat(pos.x, pos.y, pos.z);
    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(w.x, w.y, w.z);

    const auto headPos = camera.getPropertyOrValue<std::array<double, 3>>(CONTEXT_CAMERA_HEAD_POSITION, {{0., 0., 0.}});
    const auto headRotation =
        camera.getPropertyOrValue<std::array<double, 4>>(CONTEXT_CAMERA_HEAD_ROTATION, {{0., 0., 0., 1.}});
    const auto headUVec = glm::rotate(Quaterniond(headRotation[3], headRotation[0], headRotation[1], headRotation[2]),
                                      Vector3d(1., 0., 0.));

    context[CONTEXT_CAMERA_SEGMENT_ID]->setUint(camera.getPropertyOrValue<int>(CONTEXT_CAMERA_SEGMENT_ID, 0));
    context[CONTEXT_CAMERA_HALF_IPD]->setFloat(camera.getPropertyOrValue<double>(CONTEXT_CAMERA_IPD, 0.065) / 2.0);
    context[CONTEXT_CAMERA_HEAD_POSITION]->setFloat(headPos[0], headPos[1], headPos[2]);
    context[CONTEXT_CAMERA_HEAD_UVEC]->setFloat(headUVec.x, headUVec.y, headUVec.z);
    context[CONTEXT_CAMERA_APERTURE_RADIUS]->setFloat(
        camera.getPropertyOrValue<double>(CONTEXT_CAMERA_APERTURE_RADIUS, 0.0));
    context[CONTEXT_CAMERA_FOCAL_SCALE]->setFloat(
        camera.getPropertyOrValue<double>(CONTEXT_CAMERA_FOCUS_DISTANCE, 1.0));
}
} // namespace core
