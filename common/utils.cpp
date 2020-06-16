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

#include "utils.h"

namespace bioexplorer
{
std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                    std::ptr_fun<int, int>(std::isgraph)));
    return s;
}

std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::ptr_fun<int, int>(std::isgraph))
                .base(),
            s.end());
    return s;
}

std::string& trim(std::string& s)
{
    return ltrim(rtrim(s));
}

bool isClipped(const Vector3f& position, const Vector4fs& clippingPlanes)
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

void getSphericalPosition(const size_t rnd, const float assemblyRadius,
                          const PositionRandomizationType randomizationType,
                          const size_t randomSeed, const size_t occurence,
                          const size_t occurences, const Vector3f& position,
                          Vector3f& pos, Vector3f& dir)
{
    const float offset = 2.f / occurences;
    const float increment = M_PI * (3.f - sqrt(5.f));

    // Randomizer
    float radius = assemblyRadius;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        radius *= 1.f + (float(rand() % 1000 - 500) / 30000.f);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = ((occurence + rnd) % occurences) * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    dir = {x, y, z};
    pos = position + radius * dir;
}

void getFanPosition(const size_t rnd, const float assemblyRadius,
                    const PositionRandomizationType randomizationType,
                    const size_t randomSeed, const size_t occurence,
                    const size_t occurences, const Vector3f& position,
                    Vector3f& pos, Vector3f& dir)
{
    const float offset = 2.f / occurences;
    const float increment = 0.1f * M_PI * (3.f - sqrt(5.f));

    // Randomizer
    float radius = assemblyRadius;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        radius *= 1.f + (float(rand() % 1000 - 500) / 30000.f);

    // Sphere filling
    const float y = ((occurence * offset) - 1.f) + (offset / 2.f);
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = ((occurence + rnd) % occurences) * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    dir = {x, y, z};
    pos = position + radius * dir;
}

void getPlanarPosition(const float assemblyRadius,
                       const PositionRandomizationType randomizationType,
                       const size_t randomSeed, const Vector3f& position,
                       Vector3f& pos, Vector3f& dir)
{
    float up = 0.f;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        up = (float(rand() % 1000 - 500) / 20000.f);

    pos = position +
          Vector3f(float(rand() % 1000 - 500) / 1000.f * assemblyRadius, up,
                   float(rand() % 1000 - 500) / 1000.f * assemblyRadius);
    dir = {0.f, 1.f, 0.f};
}

void getCubicPosition(const float assemblyRadius, const Vector3f& position,
                      Vector3f& pos, Vector3f& dir)
{
    dir = normalize(Vector3f(float(rand() % 1000 - 500) / 1000.f,
                             float(rand() % 1000 - 500) / 1000.f,
                             float(rand() % 1000 - 500) / 1000.f));
    pos = position +
          Vector3f(float(rand() % 1000 - 500) / 1000.f * assemblyRadius,
                   float(rand() % 1000 - 500) / 1000.f * assemblyRadius,
                   float(rand() % 1000 - 500) / 1000.f * assemblyRadius);
}

float sinusoide(const float x, const float z)
{
    return 0.2f * cos(x) * sin(z) + 0.05f * cos(x * 2.3f) * sin(z * 4.6f);
}

void getSinosoidalPosition(const float size, const float height,
                           const PositionRandomizationType randomizationType,
                           const size_t randomSeed, const Vector3f& position,
                           Vector3f& pos, Vector3f& dir)
{
    const float step = 0.1f;
    const float angle = 0.1f;
    float up = 1.f;
    if (randomSeed != 0 &&
        randomizationType == PositionRandomizationType::radial)
        up = 1.f + (float(rand() % 1000 - 500) / 5000.f);

    const float x = float(rand() % 1000 - 500) / 1000.f * size;
    const float z = float(rand() % 1000 - 500) / 1000.f * size;
    const float y = height * angle * up * sinusoide(x * angle, z * angle);
    pos = Vector3f(x, y, z);

    const Vector3f v1 =
        Vector3f(x + step,
                 height * angle * up * sinusoide((x + step) * angle, z * angle),
                 z) -
        pos;
    const Vector3f v2 =
        Vector3f(x,
                 height * angle * up * sinusoide(x * angle, (z + step) * angle),
                 z + step) -
        pos;

    pos += position;
    dir = normalize(cross(normalize(v1), normalize(v2)));
}

void getBezierPosition(const Vector3fs& points, const float assemblyRadius,
                       const float t, Vector3f& pos, Vector3f& dir)
{
    Vector3fs bezierPoints = points;
    for (auto& bezierPoint : bezierPoints)
        bezierPoint *= assemblyRadius;

    size_t i = bezierPoints.size() - 1;
    while (i > 0)
    {
        for (size_t k = 0; k < i; ++k)
            bezierPoints[k] =
                bezierPoints[k] + t * (bezierPoints[k + 1] - bezierPoints[k]);
        --i;
    }
    dir = normalize(cross({0, 0, 1}, bezierPoints[1] - bezierPoints[0]));
    pos = bezierPoints[0];
}

} // namespace bioexplorer
