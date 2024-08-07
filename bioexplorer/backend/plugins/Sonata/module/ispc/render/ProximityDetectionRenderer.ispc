/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/SimulationRenderer.ih>

#include <platform/core/common/CommonTypes.h>

uniform const float nearFarThreshold = 0.2f;

struct ProximityDetectionRenderer
{
    SimulationRenderer super;

    bool surfaceShadingEnabled;

    vec3f nearColor;
    vec3f farColor;
    float detectionDistance;
    bool detectionOnDifferentMaterial;

    float alphaCorrection;
};

inline vec3f ProximityDetectionRenderer_shadeRay(const uniform ProximityDetectionRenderer* uniform self,
                                                 varying ScreenSample& sample)
{
    varying Ray ray = sample.ray;
    vec4f color = make_vec4f(0.f);
    sample.alpha = 0.f;

    int depth = 0;
    while (color.w < 1.f && depth < self->super.maxRayDepth)
    {
        traceRay(self->super.super.super.model, ray);

        if (ray.geomID < 0)
        {
            vec4f bgColor = skyboxMapping((Renderer*)self, ray, (AdvancedMaterial*)self->super.super.bgMaterial);
            bgColor.w = 1.f;
            compose(bgColor, color, self->alphaCorrection);
            return make_vec3f(color) * self->super.super.exposure;
        }

        sample.alpha = 1.f;
        if (depth == 0)
            sample.z = ray.t;

        DifferentialGeometry dg;
        postIntersect(self->super.super.super.model, dg, ray,
                      DG_NG | DG_NS | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

        uniform Material* material = dg.material;
        uniform AdvancedMaterial* advancedMaterial = (uniform AdvancedMaterial*)material;

        vec3f normal = dg.Ns;
        const vec3f P = dg.P + dg.epsilon * dg.Ng;

        bool processSurfaceShading = true;
        varying vec3f ao_dir = getRandomVector(self->super.useHardwareRandomizer, self->super.super.super.fb->size.x,
                                               sample, normal, self->super.randomNumber);

        Ray ao_ray;
        ao_ray.org = P;
        ao_ray.dir = ao_dir;
        ao_ray.t0 = max(0.f, dg.epsilon);
        ao_ray.t = self->detectionDistance;
        ao_ray.primID = -1;
        ao_ray.geomID = -1;
        ao_ray.instID = -1;

        traceRay(self->super.super.super.model, ao_ray);
        if (ao_ray.geomID != -1)
        {
            // If AO ray hits a geometry, no surface is shading required
            processSurfaceShading = false;

            DifferentialGeometry ao_dg;
            postIntersect(self->super.super.super.model, ao_dg, ao_ray, DG_MATERIALID);

            const bool doDetectionTest = self->detectionOnDifferentMaterial ? material != ao_dg.material : true;
            if (doDetectionTest)
            {
                const float a = ao_ray.t / self->detectionDistance;
                const vec4f touchColor = make_vec4f(a > nearFarThreshold ? self->nearColor : self->farColor, 1.f);
                compose(touchColor, color, self->alphaCorrection);
                sample.alpha = 1.f;
                if (depth == 0)
                    sample.z = ray.t;
            }
        }

        if (processSurfaceShading && self->surfaceShadingEnabled)
        {
            MaterialShadingMode shadingMode = undefined_shading_mode;
            vec3f Kd;
            float Ns = 1.f;
            float opacity = 1.f;
            if (!advancedMaterial)
            {
                Kd = make_vec3f(dg.color);
                opacity = dg.color.w;
            }
            else
                foreach_unique(mat in advancedMaterial)
                {
                    Kd = mat->Kd * make_vec3f(dg.color);
                    shadingMode = mat->shadingMode;
                    Ns = mat->Ns;
                    opacity = dg.color.w * mat->d;
                }

            for (uniform int i = 0; self->super.super.lights && i < self->super.super.numLights; i++)
            {
                const uniform Light* uniform light = self->super.super.lights[i];

                const vec2f samplingLocation = make_vec2f(0.f);
                const Light_SampleRes lightSample = light->sample(light, dg, samplingLocation);
                const float radiance = reduce_max(lightSample.weight);

                if (reduce_max(radiance) > 0.f)
                {
                    // Head-light shading
                    float cosNL = (shadingMode == undefined_shading_mode ? 1.f : max(0.f, dot(neg(ray.dir), dg.Ns)));

                    switch (shadingMode)
                    {
                    case electron:
                    case electron_transparency:
                        cosNL = 1.f - pow(cosNL, Ns);
                        opacity *= (shadingMode == electron_transparency ? cosNL : 1.f);
                        break;
                    case diffuse_transparency:
                        opacity *= cosNL;
                        break;
                    case cartoon:
                        const unsigned int angleAsInt = cosNL * Ns;
                        cosNL = (float)(angleAsInt) / Ns;
                        break;
                    case perlin:
                        const vec3f point = ray.org + ray.t * ray.dir;
                        float n = noise(point);
                        if (ray.t < dg.epsilon * 1.5e4f)
                            n += worleyNoise(point, 2.f);
                        else
                            n *= 2.f;

                        n = clamp(n, 0.f, 1.f);
                        cosNL *= n;
                        break;
                    }

                    const vec4f shadedColor = make_vec4f(Kd * cosNL * radiance, 1.f);
                    compose(shadedColor, color, self->alphaCorrection);
                }
            }
            break;
        }

        ray.t0 = max(0.f, ray.t + dg.epsilon);
        ray.t = inf;
        ray.primID = -1;
        ray.geomID = -1;
        ray.instID = -1;
        ++depth;
    }

    return make_vec3f(color) * self->super.super.exposure;
}

void ProximityDetectionRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                             varying ScreenSample& sample)
{
    uniform ProximityDetectionRenderer* uniform self = (uniform ProximityDetectionRenderer * uniform) _self;
    sample.ray.time = self->super.super.timestamp;
    if (self->super.super.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->super.super.anaglyphIpdOffset;
        const vec3f colorLeft = ProximityDetectionRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->super.super.anaglyphIpdOffset;
        const vec3f colorRight = ProximityDetectionRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = ProximityDetectionRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform ProximityDetectionRenderer_create(void* uniform cppE)
{
    uniform ProximityDetectionRenderer* uniform self = uniform new uniform ProximityDetectionRenderer;

    Renderer_Constructor(&self->super.super.super, cppE);
    self->super.super.super.renderSample = ProximityDetectionRenderer_renderSample;
    return self;
}

export void ProximityDetectionRenderer_set(void* uniform _self, void* uniform bgMaterial,
                                           const uniform vec3f& nearColor, const uniform vec3f& farColor,
                                           const uniform float detectionDistance,
                                           const uniform bool detectionOnDifferentMaterial,
                                           const uniform uint32 randomNumber, const uniform float timestamp,
                                           const uniform uint32 spp, const uniform bool surfaceShadingEnabled,
                                           void** uniform lights, uniform uint32 numLights,
                                           const uniform float alphaCorrection, const uniform uint32 maxRayDepth,
                                           const uniform float exposure, const uniform bool useHardwareRandomizer,
                                           const uniform bool anaglyphEnabled, const uniform vec3f& anaglyphIpdOffset)
{
    uniform ProximityDetectionRenderer* uniform self = (uniform ProximityDetectionRenderer * uniform) _self;

    // OSPRay
    self->super.super.super.spp = spp;

    // Abstract Renderer
    self->super.super.anaglyphEnabled = anaglyphEnabled;
    self->super.super.anaglyphIpdOffset = anaglyphIpdOffset;
    self->super.super.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;
    self->super.super.lights = (const uniform Light* uniform* uniform)lights;
    self->super.super.numLights = numLights;
    self->super.super.timestamp = timestamp;
    self->super.super.exposure = exposure;

    // Simulation renderer
    self->super.maxRayDepth = maxRayDepth;
    self->super.useHardwareRandomizer = useHardwareRandomizer;
    self->super.randomNumber = randomNumber;

    // Proximity detection
    self->surfaceShadingEnabled = surfaceShadingEnabled;
    self->nearColor = nearColor;
    self->farColor = farColor;
    self->detectionDistance = detectionDistance;
    self->detectionOnDifferentMaterial = detectionOnDifferentMaterial;
    self->alphaCorrection = alphaCorrection;
}
