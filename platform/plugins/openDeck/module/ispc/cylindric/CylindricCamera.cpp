/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
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

    ::ispc::CylindricCamera_set(getIE(), (const ::ispc::vec3f&)pos, (const ::ispc::vec3f&)dir, (const ::ispc::vec3f&)dir_du,
                              (const ::ispc::vec3f&)dir_dv, imgPlane_size_y);
}

OSP_REGISTER_CAMERA(CylindricCamera, cylindric);
} // namespace ospray
