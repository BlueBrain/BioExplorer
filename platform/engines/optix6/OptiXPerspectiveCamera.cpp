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

#include "OptiXPerspectiveCamera.h"

#include "OptiXCamera.h"
#include "OptiXContext.h"

#include <platform/core/common/CommonTypes.h>

#include <platform/engines/optix6/OptiX6Engine_generated_Constantbg.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_PerspectiveCamera.cu.ptx.h>

const std::string PTX_PERSPECTIVE_CAMERA = OptiX6Engine_generated_PerspectiveCamera_cu_ptx;
const std::string PTX_MISS = OptiX6Engine_generated_Constantbg_cu_ptx;

namespace core
{
OptiXPerspectiveCamera::OptiXPerspectiveCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_PERSPECTIVE_CAMERA, CUDA_FUNC_PERSPECTIVE_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_PERSPECTIVE_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXPerspectiveCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    auto position = camera.getPosition();
    const auto stereo = camera.getPropertyOrValue<bool>(CONTEXT_CAMERA_STEREO, false);
    const auto interpupillaryDistance = camera.getPropertyOrValue<double>(CONTEXT_CAMERA_IPD, DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    auto aspect = camera.getPropertyOrValue<double>(CONTEXT_CAMERA_ASPECT, 1.0);

    if (stereo)
        aspect *= 2.f;

    const auto up = glm::rotate(camera.getOrientation(), UP_VECTOR);

    Vector3d u, v, w;
    float ulen, vlen, wlen;
    w = camera.getTarget() - position;

    wlen = glm::length(w);
    u = normalize(glm::cross(w, up));
    v = normalize(glm::cross(u, w));

    vlen = wlen * tanf(0.5f * camera.getPropertyOrValue<double>(CONTEXT_CAMERA_FOVY, 45.0) * M_PI / 180.f);
    v *= vlen;
    ulen = vlen * aspect;
    u *= ulen;
    const Vector3f ipd_offset = 0.5f * interpupillaryDistance * u;

    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(w.x, w.y, w.z);

    context[CONTEXT_CAMERA_EYE]->setFloat(position.x, position.y, position.z);
    context[CONTEXT_CAMERA_APERTURE_RADIUS]->setFloat(
        camera.getPropertyOrValue<double>(CONTEXT_CAMERA_APERTURE_RADIUS, 0.0));
    context[CONTEXT_CAMERA_FOCAL_SCALE]->setFloat(camera.getPropertyOrValue<double>(CONTEXT_CAMERA_FOCAL_SCALE, 1.0));
    context[CONTEXT_CAMERA_OFFSET]->setFloat(0, 0);

    context[CONTEXT_CAMERA_STEREO]->setUint(stereo);
    context[CONTEXT_CAMERA_IPD_OFFSET]->setFloat(ipd_offset.x, ipd_offset.y, ipd_offset.z);
}
} // namespace core
