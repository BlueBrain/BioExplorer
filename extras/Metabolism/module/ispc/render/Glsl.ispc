/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue Brain Project / EPFL
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

#include "Glsl.ih"

float mix(const float x, const float y, const float a)
{
    return x * (1.f - a) + y * a;
}

vec3f mod(const vec3f& v, const int m)
{
    return make_vec3f(v.x - m * floor(v.x / m), v.y - m * floor(v.y / m),
                      v.z - m * floor(v.z / m));
}

vec3f mod(const vec3f& v, const vec3f& u)
{
    return v - u * floor(v / u);
}

float smoothstep(const float edge0, const float edge1, float x)
{
    float t;
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float sign(const float v)
{
    return v > 0 ? 1.f : v == 0.f ? 0.f : -1.f;
}

vec3f sign(const vec3f& v)
{
    return make_vec3f(sign(v.x), sign(v.y), sign(v.z));
}

// hash and noise by iq
float hash(float n)
{
    return frac(sin(n) * 43758.5453123);
}
