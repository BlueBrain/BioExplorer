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

#include "OptiXAnaglyphCamera.h"

#include "OptiXCamera.h"
#include "OptiXContext.h"

#include <platform/core/common/CommonTypes.h>

#include <platform/engines/optix6/OptiX6Engine_generated_AnaglyphCamera.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Constantbg.cu.ptx.h>

const std::string PTX_PERSPECTIVE_CAMERA = OptiX6Engine_generated_AnaglyphCamera_cu_ptx;
const std::string PTX_MISS = OptiX6Engine_generated_Constantbg_cu_ptx;

namespace core
{
OptiXAnaglyphCamera::OptiXAnaglyphCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_PERSPECTIVE_CAMERA, CUDA_FUNC_PERSPECTIVE_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, OPTIX_CUDA_FUNCTION_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_PERSPECTIVE_CAMERA, OPTIX_CUDA_FUNCTION_EXCEPTION);
}

void OptiXAnaglyphCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    auto position = camera.getPosition();
    const auto stereo = camera.getPropertyOrValue<bool>(CAMERA_PROPERTY_STEREO.name.c_str(), DEFAULT_CAMERA_STEREO);
    const auto interpupillaryDistance =
        camera.getPropertyOrValue<double>(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(),
                                          DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    auto aspect =
        camera.getPropertyOrValue<double>(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);

    if (stereo)
        aspect *= 2.f;

    const auto up = glm::rotate(camera.getOrientation(), UP_VECTOR);

    Vector3d u, v, w;
    float ulen, vlen, wlen;
    w = camera.getTarget() - position;

    wlen = glm::length(w);
    u = normalize(glm::cross(w, up));
    v = normalize(glm::cross(u, w));

    vlen = wlen * tanf(0.5f *
                       camera.getPropertyOrValue<double>(CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str(),
                                                         DEFAULT_CAMERA_FIELD_OF_VIEW) *
                       M_PI / 180.f);
    v *= vlen;
    ulen = vlen * aspect;
    u *= ulen;
    const Vector3f ipd_offset = 0.5f * interpupillaryDistance * u;

    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(w.x, w.y, w.z);

    context[CONTEXT_CAMERA_EYE]->setFloat(position.x, position.y, position.z);
    context[CONTEXT_CAMERA_APERTURE_RADIUS]->setFloat(
        camera.getPropertyOrValue<double>(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(),
                                          DEFAULT_CAMERA_APERTURE_RADIUS));
    context[CONTEXT_CAMERA_FOCAL_DISTANCE]->setFloat(
        camera.getPropertyOrValue<double>(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE));
    context[CONTEXT_CAMERA_OFFSET]->setFloat(0, 0);

    context[CONTEXT_CAMERA_STEREO]->setUint(stereo);
    context[CONTEXT_CAMERA_IPD_OFFSET]->setFloat(ipd_offset.x, ipd_offset.y, ipd_offset.z);
}
} // namespace core
