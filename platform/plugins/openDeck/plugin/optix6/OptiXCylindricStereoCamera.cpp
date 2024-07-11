/*
    Copyright 2019 - 0211 Blue Brain Project / EPFL

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

#include "OptiXCylindricStereoCamera.h"

#include <platform/engines/optix6/OptiXCamera.h>
#include <platform/engines/optix6/OptiXContext.h>

#include <CorePluginOpenDeck_generated_CylindricStereoCamera.cu.ptx.h>

#include <platform/engines/optix6/OptiX6Engine_generated_Constantbg.cu.ptx.h>

namespace core
{
namespace engine
{
namespace optix
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
    _missProgram = context->createProgramFromPTXString(PTX_MISS, OPTIX_CUDA_FUNCTION_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_CYLINDRIC_STEREO_CAMERA, OPTIX_CUDA_FUNCTION_EXCEPTION);
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
    context[CONTEXT_CAMERA_FOCAL_DISTANCE]->setFloat(
        camera.getPropertyOrValue<double>(CONTEXT_CAMERA_FOCUS_DISTANCE, 1.0));
}
} // namespace optix
} // namespace engine
} // namespace core