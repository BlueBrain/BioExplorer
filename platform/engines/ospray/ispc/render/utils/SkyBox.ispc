/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "SkyBox.ih"

vec4f skyboxMapping(const uniform Renderer* uniform renderer, const varying Ray& ray,
                    const uniform AdvancedMaterial* uniform bgMaterial)
{
    varying vec4f result = make_vec4f(bgMaterial->Kd);
    if (!valid(bgMaterial->map_Kd))
        return result;

    Ray infiniteRay = ray;

    const varying float radius = 1e6;
    infiniteRay.t = inf;

    // solve the equation sphere-ray to find the intersections
    const varying float a = 2.f * dot(infiniteRay.dir, infiniteRay.dir);
    const varying float b = 2.f * dot(infiniteRay.org, infiniteRay.dir);
    const varying float c = dot(infiniteRay.org, infiniteRay.org) - radius * radius;
    const varying float d = b * b - 2.f * a * c;

    if (d <= 0.f || a == 0.f)
        return result;

    const varying float r = sqrt(d);
    const varying float t1 = (-b - r) / a;
    const varying float t2 = (-b + r) / a;

    const varying float epsilon = 0.00001f;
    if (t1 <= epsilon && t2 <= epsilon)
        return result; // both intersections are behind the ray origin

    varying float t = 0.f;
    if (t1 <= epsilon)
        t = t2;
    else if (t2 <= epsilon)
        t = t1;
    else
        t = (t1 < t2) ? t1 : t2;

    if (t < epsilon || t > infiniteRay.t)
        return result; // Too close to intersection

    const varying vec3f intersection = normalize(infiniteRay.org + t * infiniteRay.dir);

    // Only st needs to be set to sample from Texture2D
    varying DifferentialGeometry dg;
    dg.st = make_vec2f(((atan2(-intersection.x, intersection.z) / PI) + 1.f) * .5f, (asin(intersection.y) / PI) + .5f);

    return get4f(bgMaterial->map_Kd.map, dg);
}
