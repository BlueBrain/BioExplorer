/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "Cloud.ih"
#include "Glsl.ih"

float cloudNoise(const vec3f& x)
{
    vec3f p = floor(x);
    vec3f f = frac(x);
    f = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return 2.0f * mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                          mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
                      mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                          mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y),
                      f.z) -
           1.0f;
}

float fbm(vec3f& pos, int layers, float AM, float FM)
{
    float sum = 0.0f;
    float amplitude = 1.0f;
    for (int i = 0; i < layers; ++i)
    {
        sum += amplitude * cloudNoise(pos);
        amplitude *= AM;
        pos = pos * FM;
    }
    return sum;
}

float cloud(const vec3f& p, const float time)
{
    return fbm(0.9f * make_vec3f(0.2f, 0.2f, 0.3f) *
                   (p + make_vec3f(0.0f, 0.0f, 3.0f * time)),
               7, 0.5f, 4.0f);
}
