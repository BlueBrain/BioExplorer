/*
    Copyright 2019 - 2024 Blue Brain Project / EPFL

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
namespace engine
{
namespace optix
{
OptiXOrthographicCamera::OptiXOrthographicCamera()
    : OptiXCameraProgram()
{
    auto context = OptiXContext::get().getOptixContext();
    _rayGenerationProgram = context->createProgramFromPTXString(PTX_ORTHOGRAPHIC_CAMERA, CUDA_FUNC_ORTHOGRAPHIC_CAMERA);
    _missProgram = context->createProgramFromPTXString(PTX_MISS, OPTIX_CUDA_FUNCTION_CAMERA_ENVMAP_MISS);
    _exceptionProgram = context->createProgramFromPTXString(PTX_ORTHOGRAPHIC_CAMERA, OPTIX_CUDA_FUNCTION_EXCEPTION);
}

void OptiXOrthographicCamera::commit(const OptiXCamera& camera, ::optix::Context context)
{
    const auto position = camera.getPosition();
    const auto target = camera.getTarget();
    const auto orientation = camera.getOrientation();

    const auto height = camera.getPropertyOrValue<double>(CAMERA_PROPERTY_HEIGHT.name.c_str(), DEFAULT_CAMERA_HEIGHT);
    const auto aspect =
        camera.getPropertyOrValue<double>(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);

    const Vector3d dir = normalize(target - position);

    Vector3d u = normalize(cross(dir, orientation * UP_VECTOR));
    Vector3d v = cross(u, dir);

    u *= height * aspect;
    v *= height;

    context[CONTEXT_CAMERA_U]->setFloat(u.x, u.y, u.z);
    context[CONTEXT_CAMERA_V]->setFloat(v.x, v.y, v.z);
    context[CONTEXT_CAMERA_W]->setFloat(position.x, position.y, position.z);
    context[CONTEXT_CAMERA_DIR]->setFloat(dir.x, dir.y, dir.z);
}
} // namespace optix
} // namespace engine
} // namespace core