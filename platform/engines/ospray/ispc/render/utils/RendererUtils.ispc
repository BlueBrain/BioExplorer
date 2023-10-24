/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include "RendererUtils.ih"

/**
    Composes source and destination colors according to specified alpha correction
    @param src Source RGBA color
    @param dst Returned destination RGBA color
    @param alphaCorrection Alpha correction used to blend colors
*/
inline void compose(const varying vec4f& src, varying vec4f& dst, const varying float alphaCorrection)
{
    const float alpha = alphaCorrection * src.w;
    dst = make_vec4f((1.f - dst.w) * alpha * make_vec3f(src) + dst.w * make_vec3f(dst), dst.w + alpha * (1.f - dst.w));
}

/**
   Returns the refracted vector according to the direction of the incident ray,
   he normal to the surface, and localRefraction indices
   @param direction Direction of the incident ray
   @param normal Normal to the surface
   @param n1 In localRefraction index
   @param n2 Out localRefraction index
   @return Refracted vector
*/
inline vec3f refractedVector(const varying vec3f& direction, const varying vec3f& normal, const varying float n1,
                             const varying float n2)
{
    if (n2 == 0.f)
        return direction;
    const float eta = n1 / n2;
    const float cos1 = -dot(direction, normal);
    const float cos2 = 1.f - eta * eta * (1.f - cos1 * cos1);
    if (cos2 > 0.f)
        return eta * direction + (eta * cos1 - sqrt(cos2)) * normal;
    return direction;
}

/**
    Composes source and destination colors according to specified alpha
   correction
    @param src Source RGBA color
    @param dst Returned destination RGBA color
    @param alphaCorrection Alpha correction used to blend colors
*/
inline void adaptiveCompose(const varying vec4f& src, varying vec4f& dst, const varying float alphaCorrection)
{
    const float alpha = 1.f - pow(1.f - min(src.w, 1.f - 1.f / 256.f), alphaCorrection);
    const float a = alpha * (1.f - dst.w);
    dst.x = dst.x + src.x * a;
    dst.y = dst.y + src.y * a;
    dst.z = dst.z + src.z * a;
    dst.w += (alpha * (1.f - dst.w));
}
