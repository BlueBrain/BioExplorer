/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

// Brayns
#include <brayns/ispc/render/AdvancedMaterial.ih>

// OSPRay
#include <ospray/SDK/render/Renderer.ih>
#include <ospray/SDK/render/util.ih>

vec4f skyboxMapping(const uniform Renderer* uniform renderer,
                    const varying Ray& ray,
                    const uniform AdvancedMaterial* uniform bgMaterial)
{
    if (!bgMaterial)
        return make_vec4f(1.f, 0.f, 0.f, 1.f);
    varying vec4f result = make_vec4f(bgMaterial->Kd);

    const varying float radius = 1e6;
    if (!valid(bgMaterial->map_Kd))
        return result;

    // solve the equation sphere-ray to find the intersections
    const varying float a = 2.f * dot(ray.dir, ray.dir);
    const varying float b = 2.f * dot(ray.org, ray.dir);
    const varying float c = dot(ray.org, ray.org) - radius * radius;
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

    if (t < epsilon || t > ray.t)
        return result; // Too close to intersection

    const varying vec3f intersection = normalize(ray.org + t * ray.dir);

    // Only st needs to be set to sample from Texture2D
    varying DifferentialGeometry dg;
    dg.st =
        make_vec2f(((atan2(-intersection.x, intersection.z) / PI) + 1.f) * .5f,
                   (asin(intersection.y) / PI) + .5f);

    return get4f(bgMaterial->map_Kd.map, dg);
}

void adaptiveCompose(const varying vec4f& src, varying vec4f& dst,
                     const varying float alphaCorrection)
{
    const float alpha =
        1.f - pow(1.f - min(src.w, 1.f - 1.f / 256.f), alphaCorrection);
    const float a = alpha * (1.f - dst.w);
    dst.x = dst.x + src.x * a;
    dst.y = dst.y + src.y * a;
    dst.z = dst.z + src.z * a;
    dst.w += (alpha * (1.f - dst.w));
}
