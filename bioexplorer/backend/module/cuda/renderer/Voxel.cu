/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/renderer/TransferFunction.cuh>

rtDeclareVariable(float, alphaCorrection, , );
rtDeclareVariable(float, simulationThreshold, , );

static __device__ inline void shade()
{
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 color = make_float3(0.f);
    if (prd.depth < maxBounces && cast_user_data && simulation_data.size() > 0)
    {
        const float4 userDataColor = calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange,
                                                               simulation_data[simulation_idx], tfColors, tfOpacities);

        if (userDataColor.w >= simulationThreshold)
        {
            color = color * (1.f - userDataColor.w) + make_float3(userDataColor) * userDataColor.w;
            prd.importance = userDataColor.w * alphaCorrection;
        }
        else
            prd.importance = 0.f;

        PerRayData_radiance new_prd;
        new_prd.depth = prd.depth + 1;

        const optix::Ray new_ray = optix::make_Ray(hit_point, ray.direction, radianceRayType, sceneEpsilon, ray.tmax);
        rtTrace(top_object, new_ray, new_prd);
    }

    color = ::optix::clamp(mainExposure * color, 0.f, 1.f);
    prd.result = color;
}

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade();
}
