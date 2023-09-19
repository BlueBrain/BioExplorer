// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "OrthographicCamera.h"
#include "OrthographicCamera_ispc.h"

#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/Data.h>

using namespace core;

namespace ospray
{
OrthographicCamera::OrthographicCamera()
{
    ispcEquivalent = ispc::OrthographicCamera_create(this);
}

void OrthographicCamera::commit()
{
    Camera::commit();

    height = getParamf(CAMERA_PROPERTY_HEIGHT.name.c_str(), DEFAULT_CAMERA_HEIGHT);
    aspect = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);
    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;

    dir = normalize(dir);
    vec3f pos_du = normalize(cross(dir, up));
    vec3f pos_dv = cross(pos_du, dir);

    pos_du *= height * aspect;
    pos_dv *= height;

    vec3f pos_00 = pos - 0.5f * pos_du - 0.5f * pos_dv;

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    ispc::OrthographicCamera_set(getIE(), (const ispc::vec3f&)dir, (const ispc::vec3f&)pos_00,
                                 (const ispc::vec3f&)pos_du, (const ispc::vec3f&)pos_dv,
                                 (const ispc::vec4f*)clipPlaneData, numClipPlanes);
}

OSP_REGISTER_CAMERA(OrthographicCamera, orthographic);

} // namespace ospray
