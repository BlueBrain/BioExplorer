/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "CylindricCamera.h"
#include "CylindricCamera_ispc.h"

namespace
{
constexpr float OPENDECK_FOV_Y = 48.549f;
}

namespace ospray
{
CylindricCamera::CylindricCamera()
{
    ispcEquivalent = ::ispc::CylindricCamera_create(this);
}

std::string CylindricCamera::toString() const
{
    return "ospray::CylindricCamera";
}

void CylindricCamera::commit()
{
    Camera::commit();

    dir = normalize(dir);
    const auto dir_du = normalize(cross(dir, up));
    const auto dir_dv = normalize(up);
    dir = -dir;

    const auto imgPlane_size_y = 2.0f * tanf(deg2rad(0.5f * OPENDECK_FOV_Y));

    ::ispc::CylindricCamera_set(getIE(), (const ::ispc::vec3f&)pos, (const ::ispc::vec3f&)dir,
                                (const ::ispc::vec3f&)dir_du, (const ::ispc::vec3f&)dir_dv, imgPlane_size_y);
}

OSP_REGISTER_CAMERA(CylindricCamera, cylindric);
} // namespace ospray
