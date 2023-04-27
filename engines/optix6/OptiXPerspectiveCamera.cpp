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

#include "OptiXPerspectiveCamera.h"

#include "OptiXCamera.h"
#include "OptiXContext.h"

#include <brayns/common/CommonTypes.h>

#include <engines/optix6/braynsOptix6Engine_generated_Constantbg.cu.ptx.h>
#include <engines/optix6/braynsOptix6Engine_generated_PerspectiveCamera.cu.ptx.h>

const std::string CUDA_PERSPECTIVE_CAMERA = braynsOptix6Engine_generated_PerspectiveCamera_cu_ptx;
const std::string CUDA_MISS = braynsOptix6Engine_generated_Constantbg_cu_ptx;

const std::string CUDA_FUNC_PERSPECTIVE_CAMERA = "perspectiveCamera";
const std::string CUDA_ATTR_CAMERA_APERTURE_RADIUS = "apertureRadius";
const std::string CUDA_ATTR_CAMERA_FOCAL_SCALE = "focusDistance";
const std::string CUDA_ATTR_CAMERA_FOVY = "fovy";
const std::string CUDA_ATTR_CAMERA_STEREO = "stereo";
const std::string CUDA_ATTR_CAMERA_INTERPUPILLARY_DISTANCE = "interpupillaryDistance";
const std::string CUDA_ATTR_CAMERA_EYE_OFFSET = "ipd_offset";

namespace brayns
{
OptiXPerspectiveCamera::OptiXPerspectiveCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(CUDA_PERSPECTIVE_CAMERA, CUDA_FUNC_PERSPECTIVE_CAMERA);
    _missProgram = context->createProgramFromPTXString(CUDA_MISS, CUDA_FUNC_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(CUDA_PERSPECTIVE_CAMERA, CUDA_FUNC_CAMERA_EXCEPTION);
}

void OptiXPerspectiveCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    auto position = camera.getPosition();
    const auto stereo = camera.getPropertyOrValue<bool>(CUDA_ATTR_CAMERA_STEREO, false);
    const auto interpupillaryDistance =
        camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_INTERPUPILLARY_DISTANCE, 0.0635f);
    auto aspect = camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_ASPECT, 1.0);

    if (stereo)
        aspect *= 2.f;

    const auto up = glm::rotate(camera.getOrientation(), Vector3d(0, 1, 0));

    Vector3d u, v, w;
    float ulen, vlen, wlen;
    w = camera.getTarget() - position;

    wlen = glm::length(w);
    u = normalize(glm::cross(w, up));
    v = normalize(glm::cross(u, w));

    vlen = wlen * tanf(0.5f * camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_FOVY, 45.0) * M_PI / 180.f);
    v *= vlen;
    ulen = vlen * aspect;
    u *= ulen;
    const Vector3f ipd_offset = 0.5f * interpupillaryDistance * u;

    context[CUDA_ATTR_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CUDA_ATTR_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CUDA_ATTR_CAMERA_W]->setFloat(w.x, w.y, w.z);

    context[CUDA_ATTR_CAMERA_EYE]->setFloat(position.x, position.y, position.z);
    context[CUDA_ATTR_CAMERA_APERTURE_RADIUS]->setFloat(
        camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_APERTURE_RADIUS, 0.0));
    context[CUDA_ATTR_CAMERA_FOCAL_SCALE]->setFloat(
        camera.getPropertyOrValue<double>(CUDA_ATTR_CAMERA_FOCAL_SCALE, 1.0));
    context[CUDA_ATTR_CAMERA_BAD_COLOR]->setFloat(1.f, 0.f, 1.f);
    context[CUDA_ATTR_CAMERA_OFFSET]->setFloat(0, 0);

    context[CUDA_ATTR_CAMERA_STEREO]->setUint(stereo);
    context[CUDA_ATTR_CAMERA_EYE_OFFSET]->setFloat(ipd_offset.x, ipd_offset.y, ipd_offset.z);
}
} // namespace brayns