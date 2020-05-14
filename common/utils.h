/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_UTILS_H
#define BIOEXPLORER_UTILS_H

#include <brayns/common/types.h>

namespace bioexplorer
{
inline std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::ptr_fun<int, int>(std::isgraph)));
    return s;
}

inline std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::ptr_fun<int, int>(std::isgraph))
                .base(),
            s.end());
    return s;
}

inline std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

inline bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes)
{
    if (clippingPlanes.empty())
        return false;

    bool visible = true;
    for (auto plane : clippingPlanes)
    {
        const Vector3f normal = {plane.x, plane.y, plane.z};
        const float d = plane.w;
        const float distance = dot(normal, position) + d;
        visible &= (distance > 0.f);
    }
    return !visible;
}

inline void getSphericalPosition(
    const size_t rnd, const float assemblyRadius,
    const PositionRandomizationType randomizationType, const size_t randomSeed,
    const size_t occurence, const size_t occurences, Vector3f& position,
    Vector3f& direction)
{
    const float offset = 2.f / occurences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    // Randomizer
    float radius = assemblyRadius;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        radius *= 1.f + (float(rand() % 1000 - 500) / 20000.f);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = ((occurence + rnd) % occurences) * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    direction = {x, y, z};
    position = radius * direction;
}

inline void getPlanarPosition(const float assemblyRadius,
                              const PositionRandomizationType randomizationType,
                              const size_t randomSeed, Vector3f& position,
                              Vector3f& direction)
{
    // Randomizer
    float up = 0.f;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        up = (float(rand() % 1000 - 500) / 20000.f);

    position = {float(rand() % 1000 - 500) / 1000.f * assemblyRadius, up,
                float(rand() % 1000 - 500) / 1000.f * assemblyRadius};
}

} // namespace bioexplorer

#endif // BIOEXPLORER_UTILS_H
