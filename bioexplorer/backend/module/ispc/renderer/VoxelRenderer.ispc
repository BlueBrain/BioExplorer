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

#include "VoxelRenderer.ih"

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/SkyBox.ih>

void linearCompose(const varying vec4f& src, varying vec4f& dst, const varying float alphaRatio)
{
    const float a = alphaRatio * src.w;
    dst = make_vec4f((1.f - dst.w) * a * make_vec3f(src) + dst.w * make_vec3f(dst), dst.w + a);
}

inline vec3f VoxelRenderer_shadeRay(const uniform VoxelRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    ray.time = inf;
    sample.z = inf;

    float zDepth = 0.f;
    int depth = 0;

    vec4f pathColor = make_vec4f(0.f);

    while (pathColor.w < 1.f)
    {
        traceRay(self->super.super.super.model, ray);

        if (ray.geomID < 0)
        {
            // No intersection
            const vec4f backgroundColor =
                skyboxMapping((Renderer*)self, ray, (AdvancedMaterial*)self->super.super.bgMaterial);
            pathColor =
                make_vec4f(make_vec3f(backgroundColor) * (1.f - pathColor.w) + make_vec3f(pathColor) * pathColor.w,
                           pathColor.w);
            break;
        }

        DifferentialGeometry dg;
        postIntersect(self->super.super.super.model, dg, ray, DG_FACEFORWARD | DG_MATERIALID);

        uniform Material* material = dg.material;
        uniform AdvancedMaterial* objMaterial = (uniform AdvancedMaterial*)material;

        bool castUserData = false;
        if (objMaterial)
            foreach_unique(mat in objMaterial) castUserData = mat->castUserData;

        vec4f colorContribution = make_vec4f(0.f);
        if (castUserData && self->super.simulationDataSize > 0)
            colorContribution = getSimulationValue(&self->super, &dg, ray.primID);

        if (colorContribution.w > self->simulationThreshold)
        {
            adaptiveCompose(colorContribution, pathColor, self->super.alphaCorrection);
            // Depth
            if (depth == 0)
                sample.z = ray.t;

            ++depth;
        }

        // Next ray
        ray.t0 = max(0.f, ray.t + dg.epsilon);
        ray.t = inf;
        ray.primID = -1;
        ray.geomID = -1;
        ray.instID = -1;
    }

    // Alpha
    sample.alpha = pathColor.w;

    return make_vec3f(pathColor) * self->super.super.exposure;
}

void VoxelRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                varying ScreenSample& sample)
{
    uniform VoxelRenderer* uniform self = (uniform VoxelRenderer * uniform) _self;
    sample.ray.time = self->super.super.timestamp;
    if (self->super.super.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->super.super.anaglyphIpdOffset;
        const vec3f colorLeft = VoxelRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->super.super.anaglyphIpdOffset;
        const vec3f colorRight = VoxelRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = VoxelRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform VoxelRenderer_create(void* uniform cppE)
{
    uniform VoxelRenderer* uniform self = uniform new uniform VoxelRenderer;
    Renderer_Constructor(&self->super.super.super, cppE);
    self->super.super.super.renderSample = VoxelRenderer_renderSample;
    return self;
}

export void VoxelRenderer_set(void* uniform _self, void* uniform bgMaterial, const uniform uint32 spp,
                              uniform float* uniform simulationData, const uniform uint64 simulationDataSize,
                              const uniform float alphaCorrection, const uniform float simulationThreshold,
                              const uniform float exposure, const uniform bool anaglyphEnabled,
                              const uniform vec3f& anaglyphIpdOffset)
{
    uniform VoxelRenderer* uniform self = (uniform VoxelRenderer * uniform) _self;

    // OSPRay
    self->super.super.super.spp = spp;

    // Abstract
    self->super.super.anaglyphEnabled = anaglyphEnabled;
    self->super.super.anaglyphIpdOffset = anaglyphIpdOffset;
    self->super.super.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;
    self->super.super.exposure = exposure;

    // Simulation
    self->super.simulationData = (uniform float* uniform)simulationData;
    self->super.simulationDataSize = simulationDataSize;
    self->super.alphaCorrection = alphaCorrection;

    // Voxel
    self->simulationThreshold = simulationThreshold;
}
